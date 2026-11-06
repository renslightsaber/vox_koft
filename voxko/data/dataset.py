"""학습 Dataset — 16kHz wav + "(지시문)문장" + (선택) 같은 화자 무감정 ref.

한 문장의 학습 입력 (공식 AudioFeatureProcessingPacker 가 GPU 에서 만든다)
    ref 있음 : [103, ref…, 104, "(지시문)문장", 101, target…, 102]   ← 추론 reference_wav_path 와 같은 모양
    ref 없음 : ["(지시문)문장", 101, target…, 102]
    loss 는 target 에만(loss/diff · loss/stop).

결정성: 지시문을 붙일지 · ref 를 붙일지 · 어느 ref 인지 · 어느 문구인지는 Random(f"{seed}|{epoch}|{utt}") 로 정한다.
    난수를 꺼내는 순서 = ① 지시문 여부 ② (ref 후보가 있으면) ref 여부 · 어느 ref ③ (문구가 2개 이상이면) 어느 문구.
    ③ 은 맨 뒤라서 문구가 하나뿐인 표(v1)는 2026-11-05 이전 코드와 입력이 완전히 같다(tests 로 확인).
지시문 표(configs/emotion/*.yaml): 감정마다 문구 하나(str) 또는 여러 개(list — 학습 때 하나를 고름).
지시문 빼기(data.emotion.instruct_ratio + drop_only):
    drop_only 가 없으면 모든 문장이 instruct_ratio 확률로 지시문을 붙인다(v1).
    drop_only: [Neutrality] 면 **전체의 (1 − instruct_ratio)** 만큼을 그 감정 문장에서만 뺀다
    → 그 감정 문장의 지시문 없앨 확률 = (1 − ratio) × 전체 문장 수 / 그 감정 문장 수 · 다른 감정 문장은 항상 지시문.
    → DataLoader worker 수 · GPU 수 · resume 과 무관하게 같은 epoch 의 같은 문장은 같은 입력. val(epoch 없음)은 epoch 0 고정.
고정 ref(2026-11-06 · A7NX · A7NX3 "다른 칸 ref"): 행에 ref_audio 가 있으면 그 ref 를 **항상** 씀(ref_ratio · 후보 풀 무시) ·
    길이 계산도 그 ref 길이(ref_duration) · 난수 키는 utt_x(같은 정답이 여러 행일 때 구분) — 없으면 utt. 행에 없으면 이전과 완전히 같음.
길이 거르기: 근사 길이(text + target patch + 그 화자 ref 최대 patch + 4)가 max_seq_len 을 넘으면 제외(n_dropped).
    공식 packer 는 8192 를 넘으면 말없이 잘라 stop 라벨이 사라진다.
"""
from __future__ import annotations

import collections
import json
import math
import random
import re
from typing import Callable, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 16000
PATCH_SAMPLES = 640 * 4          # AudioVAE hop 640 × patch 4 → LM 한 칸 = 0.16초
PAD = -100


def read_jsonl(p: str) -> List[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def n_patches(n_samples: int) -> int:
    """packer 의 encode_audio(→ patch 배수로 오른쪽 패딩)와 같은 계산 = loss 가 걸리는 target 칸 수."""
    return max(1, math.ceil(n_samples / PATCH_SAMPLES))


def approx_patches(dur: float) -> int:
    return n_patches(int(round(dur * SR)))


def clean_text(t: str) -> str:
    """추론(core.py: 줄바꿈 → 공백, 연속 공백 → 하나)과 같은 공백 처리."""
    return re.sub(r"\s+", " ", t.replace("\n", " ")).strip()


def load_instruct_map(path: str) -> Dict[str, List[str]]:
    """감정 → 문구 목록. 값이 str 이면 [str]. 문구 안에 괄호가 있으면 에러(공식 형식: 괄호 안에 괄호 없음)."""
    import yaml
    with open(path, encoding="utf-8") as f:
        m = (yaml.safe_load(f) or {}).get("instruct") or {}
    if not m:
        raise ValueError(f"감정 지시문 표에 instruct: 가 없습니다: {path}")
    out = {}
    for k, v in m.items():
        vs = [str(x).strip() for x in (v if isinstance(v, list) else [v])]
        if not vs or any(not x or "(" in x or ")" in x for x in vs):
            raise ValueError(f"지시문 표 {path} 의 {k}: 빈 문구 또는 괄호가 든 문구가 있습니다: {vs}")
        out[str(k)] = vs
    return out


def build_pool(pool_rows: List[dict], min_dur: float, max_dur: float) -> Dict[str, List[dict]]:
    pool = collections.defaultdict(list)
    for r in pool_rows:
        if min_dur <= float(r["duration"]) <= max_dur:
            pool[r["speaker"]].append({"utt": r["utt"], "audio": r["audio"], "duration": float(r["duration"])})
    for v in pool.values():
        v.sort(key=lambda x: x["utt"])                       # 파일 순서와 무관하게 결정적
    return dict(pool)


class VoxDataset(Dataset):
    wants_epoch = True                                       # BucketBatchSampler 가 (idx, epoch) 를 넘긴다

    def __init__(self, rows: List[dict], tokenize: Callable[[str], List[int]], instruct_map: Dict[str, List[str]],
                 instruct_ratio: float, ref_ratio: float, pool: Dict[str, List[dict]], seed: int,
                 max_seq_len: int, name: str = "train", drop_only: Optional[List[str]] = None, strict: bool = True):
        missing = sorted({r.get("emotion") for r in rows} - set(instruct_map))
        if instruct_ratio > 0 and missing:
            raise ValueError(f"[{name}] 감정 지시문 표에 없는 감정: {missing} — configs/emotion/ 표를 확인하세요")
        self.name, self.seed = name, int(seed)
        self.instruct_map = {k: [str(x) for x in (v if isinstance(v, list) else [v])] for k, v in instruct_map.items()}
        self.instruct_ratio, self.ref_ratio = float(instruct_ratio), float(ref_ratio)
        self.drop_only = list(drop_only) if drop_only else None
        self.pool = pool
        self.rows, self.ids_plain, self.ids_instr, self.lengths = [], [], [], []
        self.n_dropped, self.n_no_pool = 0, 0
        self.drop_warning = None
        ref_max = {s: max(approx_patches(x["duration"]) for x in v) for s, v in pool.items() if v}
        as_ids = lambda s: np.asarray(list(tokenize(s)), dtype=np.int32)  # noqa: E731  (메모리: list 대신 int32 배열)
        for r in rows:
            text = clean_text(r["text"])
            plain = as_ids(text)
            instr = ([as_ids(f"({p}){text}") for p in self.instruct_map[r["emotion"]]]
                     if instruct_ratio > 0 else [plain])
            cands = self._candidates(r)
            if r.get("ref_audio"):                                # 고정 ref — 항상 붙음
                ref_len = approx_patches(float(r["ref_duration"])) + 2
            else:
                ref_len = (ref_max.get(r["speaker"], 0) + 2) if (self.ref_ratio > 0 and cands) else 0
            L = max(len(plain), *(len(x) for x in instr)) + approx_patches(float(r["duration"])) + ref_len + 2
            if L > max_seq_len:
                self.n_dropped += 1
                continue
            self.n_no_pool += int(self.ref_ratio > 0 and not cands and not r.get("ref_audio"))
            self.rows.append({**r, "text": text})
            self.ids_plain.append(plain)
            self.ids_instr.append(instr)
            self.lengths.append(L)
        # 문장별 지시문 붙일 확률
        self.p_keep_drop = self.instruct_ratio                # drop_only 감정 문장의 확률(없으면 모든 문장 공통)
        if self.drop_only:
            n_all = len(self.rows)
            n_el = sum(1 for r in self.rows if r["emotion"] in self.drop_only)
            if n_el == 0:
                self.p_keep_drop = 1.0
            else:
                p_drop = (1.0 - self.instruct_ratio) * n_all / n_el
                if p_drop > 1.0 + 1e-9:
                    msg = (f"[{name}] instruct_ratio={self.instruct_ratio} 를 맞추려면 {self.drop_only} 문장 "
                           f"{n_el}/{n_all} 중 {p_drop:.2f} 배를 빼야 합니다(불가능)")
                    if strict:                                # 학습 데이터: 멈춤
                        raise ValueError(msg + " — 비율을 올리거나 drop_only 를 넓히세요")
                    self.drop_warning = msg + " → 그 감정 문장 전부에서 뺌(평가 세트라 계속)"   # 평가 세트: 가능한 만큼
                    p_drop = 1.0
                self.p_keep_drop = 1.0 - p_drop

    def p_keep(self, i: int) -> float:
        if self.drop_only is None:
            return self.instruct_ratio
        return self.p_keep_drop if self.rows[i]["emotion"] in self.drop_only else 1.0

    def instr_summary(self) -> str:
        n_ph = sorted({len(v) for v in self.instruct_map.values()})
        if self.drop_only is None:
            return f"비율 {self.instruct_ratio} (모든 문장 같음) · 감정당 문구 {n_ph}"
        return (f"전체 {self.instruct_ratio} · 빼는 것은 {self.drop_only} 문장만(그 문장의 지시문 확률 {self.p_keep_drop:.3f}) · "
                f"감정당 문구 {n_ph}" + (f" ⚠️ {self.drop_warning}" if self.drop_warning else ""))

    def _candidates(self, r: dict) -> List[dict]:
        return [x for x in self.pool.get(r["speaker"], []) if x["utt"] != r["utt"]]

    def __len__(self) -> int:
        return len(self.rows)

    def plan(self, i: int, epoch: int = 0) -> dict:
        """문장 i 의 이번 epoch 입력 구성(지시문 여부 · ref) — 결정적."""
        r = self.rows[i]
        rng = random.Random(f"{self.seed}|{epoch}|{r.get('utt_x', r['utt'])}")
        use_instr = rng.random() < self.p_keep(i)            # ① (v1 과 같은 첫 난수)
        ref = None
        if r.get("ref_audio"):                                # 고정 ref(행에 지정) — 비율 · 풀 무시, 난수 안 씀
            ref = {"utt": r["ref_utt"], "audio": r["ref_audio"], "duration": float(r["ref_duration"])}
        else:
            cands = self._candidates(r)
            if cands and rng.random() < self.ref_ratio:       # ② (v1 과 같은 순서)
                ref = cands[rng.randrange(len(cands))]
        phrases = self.instruct_map.get(r["emotion"], [""])
        k = rng.randrange(len(phrases)) if (use_instr and len(phrases) > 1) else 0   # ③ 맨 뒤 — v1(문구 1개)은 꺼내지 않음
        text = f"({phrases[k]}){r['text']}" if use_instr else r["text"]
        return {"use_instr": use_instr, "ref": ref, "text": text, "phrase": k if use_instr else None}

    @staticmethod
    def _load(path: str) -> np.ndarray:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if sr != SR:
            raise ValueError(f"16kHz 가 아닙니다({sr}): {path} — voxko.data.prepare 로 변환한 파일을 쓰세요")
        return wav.mean(axis=1) if wav.ndim > 1 else wav

    def __getitem__(self, key) -> dict:
        i, epoch = key if isinstance(key, tuple) else (key, 0)
        r, p = self.rows[i], self.plan(i, epoch)
        wav = self._load(r["audio"])
        ref = self._load(p["ref"]["audio"]) if p["ref"] else None
        ids = self.ids_instr[i][p["phrase"] if len(self.ids_instr[i]) > 1 else 0] if p["use_instr"] else self.ids_plain[i]
        return {"text_ids": ids, "audio": wav, "ref": ref,
                "utt": r.get("utt_x", r["utt"]), "ref_utt": p["ref"]["utt"] if p["ref"] else None}

    @staticmethod
    def collate(batch: List[dict]) -> dict:
        def pad(seqs, dtype, value):
            m = max(len(s) for s in seqs)
            out = torch.full((len(seqs), m), value, dtype=dtype)
            for k, s in enumerate(seqs):
                out[k, :len(s)] = torch.as_tensor(s, dtype=dtype)
            return out

        refs = [b["ref"] if b["ref"] is not None else np.zeros(0, np.float32) for b in batch]
        has_ref = any(len(x) for x in refs)
        out = {"text_tokens": pad([b["text_ids"] for b in batch], torch.int32, PAD),
               "audio_tokens": pad([b["audio"] for b in batch], torch.float32, float(PAD)),
               "counts": torch.tensor(float(sum(n_patches(len(b["audio"])) for b in batch))),
               "n_ref": torch.tensor(float(sum(1 for x in refs if len(x)))),
               "utts": [b["utt"] for b in batch], "ref_utts": [b["ref_utt"] for b in batch]}
        if has_ref:      # ref 없는 문장은 길이 0(전부 PAD) → packer 가 ref 없는 경로로 처리
            m = max(len(x) for x in refs)
            t = torch.full((len(refs), m), float(PAD), dtype=torch.float32)
            for k, x in enumerate(refs):
                t[k, :len(x)] = torch.as_tensor(x)
            out["ref_audio_tokens"] = t
        return out


def make_dataset(jsonl: str, pool_jsonl: Optional[str], tokenize, instruct_map, dc: dict, seed: int, name: str,
                 extra_pool_rows: Optional[List[dict]] = None, strict: bool = True) -> VoxDataset:
    """설정(data.*)으로 Dataset 을 만든다. pool=any 면 이 jsonl 의 문장들도 ref 후보."""
    rows = read_jsonl(jsonl)
    rf = dc["ref"]
    pool_rows = (read_jsonl(pool_jsonl) if pool_jsonl else []) + list(extra_pool_rows or [])
    if rf["pool"] == "any":
        pool_rows += rows
    pool = build_pool(pool_rows, float(rf["min_dur"]), float(rf["max_dur"]))
    return VoxDataset(rows, tokenize, instruct_map, float(dc["emotion"]["instruct_ratio"]), float(rf["ratio"]),
                      pool, seed, int(dc["max_seq_len"]), name=name, drop_only=dc["emotion"].get("drop_only"),
                      strict=strict)
