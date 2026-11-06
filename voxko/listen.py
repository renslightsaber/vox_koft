"""청음 노트북 도우미 — run 정보 · step 목록 · ref(처음 보는 화자 16명) · 합성 표(원본과 나란히).

노트북(notebooks/<sft|lora>/<run>_listening.ipynb)에서:
    from voxko import listen
    L = listen.Listener(RUN_NAME, seed=SEED, cfg_value=CFG, timesteps=TIMESTEPS)
    listen.show_refs(); ref = listen.pick_ref(0)
    L.grid(steps=[None, 750, 1500], emotions=listen.EMO7, ref=ref)      # None = 원본 VoxCPM2
입력 규칙은 D 재측정과 같다: "(v1 지시문)문장" + 목표 화자 무감정 reference · prompt 없음 · 준비 합성 뒤 · seed 고정.
LISTEN_SMOKE=1 이면 각 표를 1칸 · 1열로 줄여 빠르게 실행만 확인한다(노트북 검증용).
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

H = Path.home() / "workspaces"
OUT = H / "outputs/vox_koft"
BASE = H / "ourckpts/voxcpm/VoxCPM2"
D_EVAL = H / "tmp_listen/vox_koft/d_eval"
EMAP = H / "vox_koft/configs/emotion/a7n_vox_v1.yaml"
EMO7 = ["Neutrality", "Happy-A", "Happy-B", "Sad-A", "Sad-B", "Angry-A", "Angry-B"]
KO = {"Neutrality": "무감정", "Happy-A": "기쁨 A(약)", "Happy-B": "기쁨 B(강)", "Sad-A": "슬픔 A(약)", "Sad-B": "슬픔 B(강)",
      "Angry-A": "분노 A(약)", "Angry-B": "분노 B(강)"}
SMOKE = os.environ.get("LISTEN_SMOKE") == "1"
# 칸마다 들을 기본 문장(A7N val — 학습에 안 쓴 문장, D 재측정과 같은 것 + 무감정 문장 하나)
DEFAULT_TEXT = {"Neutrality": "오늘 회의는 오후 세 시에 이 층 회의실에서 시작합니다."}


def run_info(run: str) -> dict:
    import yaml
    m = yaml.safe_load("\n".join(l for l in open(OUT / "ckpt" / run / "configs/merged.yaml", encoding="utf-8") if not l.startswith("#")))
    d, lo = m["data"], m["lora"]
    return {"방식": "LoRA r%d" % lo["r"] if lo["enable"] else "전체 FT", "lr": m["optim"]["lr"], "max_steps": m["train"]["max_steps"],
            "지시문 표": Path(d["emotion"]["map_file"]).name, "지시문 비율": d["emotion"]["instruct_ratio"],
            "drop_only": d["emotion"].get("drop_only"), "ref 비율": d["ref"]["ratio"],
            "batch": f"{d['batch_size']} × accum {m['train']['grad_accum']} × GPU {m['hardware']['expected_num_processes']}"}


def steps(run: str) -> List[int]:
    return sorted(int(p.name.split("_")[1]) for p in (OUT / "ckpt" / run).glob("step_*") if (p / "_COMPLETE").exists())


def export_dir(run: str, step: int) -> Path:
    d = OUT / "ckpt" / run / f"step_{step}"
    for k in ("export_ft", "export_lora"):
        if (d / k).is_dir():
            return d / k
    raise FileNotFoundError(f"{d} 에 export_ft/export_lora 가 없습니다")


def refs() -> list:
    """처음 보는 화자 16명(542 여4 · 남4 + val_unseen8)의 무감정 reference — D 재측정과 같은 파일."""
    P = json.load(open(D_EVAL / "plan.json"))
    return [dict(idx=i, speaker=s, gender=v["gender"], src=v["src"], ref=v["ref"], ref_text=v["ref_text"])
            for i, (s, v) in enumerate(P["spk"].items())]


def show_refs():
    from IPython.display import HTML, display
    rows = "".join(f"<tr><td>{r['idx']}</td><td>{r['speaker']}</td><td>{'여' if r['gender'] == 'female' else '남'}</td><td>{r['src']}</td>"
                   f"<td>{_audio_file(r['ref'])}</td><td>{html.escape(r['ref_text'])}</td></tr>" for r in refs())
    display(HTML(f"<table><tr><th>#</th><th>화자</th><th>성별</th><th>출처</th><th>ref</th><th>전사문</th></tr>{rows}</table>"))


def pick_ref(i: int) -> dict:
    return refs()[i]


def texts() -> dict:
    """칸 → 문장 목록(D 재측정 문장 2개씩 · Neutrality 는 기본 문장)."""
    P = json.load(open(D_EVAL / "plan.json"))
    out = {e: [] for e in EMO7}
    for x in P["sents"]:
        out[x["emo"]].append(x["text"])
    out["Neutrality"] = [DEFAULT_TEXT["Neutrality"]]
    return out


def _b64(wav: np.ndarray, sr: int) -> str:
    import soundfile as sf
    b = io.BytesIO()
    sf.write(b, np.asarray(wav, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(b.getvalue()).decode()


def _audio(wav, sr) -> str:
    return f'<audio controls preload="none" style="width:200px" src="{_b64(wav, sr)}"></audio>'


def _audio_file(path: str) -> str:
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32")
    return _audio(y, sr)


class Listener:
    """모델 캐시(원본 + 최근 1개) · 합성 · 표. step=None 은 원본 VoxCPM2."""

    def __init__(self, run: str, seed: int = 1986, cfg_value: float = 2.0, timesteps: int = 10, max_len: int = 600):
        self.run, self.seed, self.cfg, self.ts, self.max_len = run, seed, cfg_value, timesteps, max_len
        self._eng = {}

    def engine(self, step: Optional[int]):
        from voxko.engine.synth import Engine
        key = "base" if step is None else int(step)
        if key not in self._eng:
            for k in [k for k in self._eng if k != "base"]:          # 원본 + 최근 1개만(메모리)
                del self._eng[k]
            import torch
            torch.cuda.empty_cache()
            path = BASE if step is None else export_dir(self.run, int(step))
            self._eng[key] = Engine(str(path), emotion_map=str(EMAP))     # 준비 합성 포함
        return self._eng[key]

    def synth(self, step, text: str, emotion: Optional[str] = None, ref: Optional[dict] = None, seed: Optional[int] = None,
              instruct: Optional[str] = None, normalize: bool = True):
        e = self.engine(step)
        t = e.compose(text, emotion=emotion, instruct=instruct, normalize=normalize)
        w = e.synth(t, ref=(ref or {}).get("ref"), seed=self.seed if seed is None else seed, cfg_value=self.cfg,
                    inference_timesteps=self.ts, max_len=self.max_len)
        return w, e.sr, t

    def grid(self, steps: list, emotions: list = EMO7, ref: Optional[dict] = None, text_idx: int = 0, seed: Optional[int] = None):
        """줄 = 칸(감정), 열 = step(None = 원본). 같은 문장 · ref · seed."""
        from IPython.display import HTML, display
        if SMOKE:
            steps, emotions = steps[:2], emotions[:1]
        T = texts()
        cells = {}
        for st in steps:                                              # step 하나씩 불러 모든 칸을 합성(교체 최소화)
            for emo in emotions:
                tx = T[emo][min(text_idx, len(T[emo]) - 1)]
                w, sr, full = self.synth(st, tx, emotion=emo, ref=ref, seed=seed)
                cells[(emo, st)] = (_audio(w, sr) + f"<div style='color:#666;font-size:.8em'>{len(w) / sr:.1f}초</div>", full)
        head = "".join(f"<th>{'원본' if s is None else f'step {s}'}</th>" for s in steps)
        rows = "".join(f"<tr><td>{KO[e]}<div style='color:#666;font-size:.75em'>{html.escape(cells[(e, steps[0])][1])}</div></td>"
                       + "".join(f"<td>{cells[(e, s)][0]}</td>" for s in steps) + "</tr>" for e in emotions)
        who = f"ref = {ref['speaker']}({'여' if ref['gender'] == 'female' else '남'} · {ref['src']})" if ref else "ref 없음"
        display(HTML(f"<p><b>{self.run}</b> · {who} · seed {self.seed if seed is None else seed}</p>"
                     f"<table><tr><th>칸 · 입력</th>{head}</tr>{rows}</table>"))
