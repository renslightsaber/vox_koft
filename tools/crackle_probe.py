#!/usr/bin/env python
"""갈라짐(거친 결 · 삑사리) 지표 시험 — 사용자 청음 메모로 어떤 숫자가 갈라짐을 가려내는지 본다 (2026-11-09 · 지시문 14:00 §2 고르기 규칙 1)

정답(사용자 메모 · B 4모델 페이지 · C sNp 페이지 · 11/08 ~ 11/09): 줄(화자 · 칸 · 문장) × 모델 중 메모에 "갈라짐 · 삑사리 · 클리핑"이
그 모델(또는 "네 개 다" · "비교군 빼고 나머지")에 적힌 것 = 1, 같은 페이지 B칸(기쁨B · 슬픔B · 분노B) 나머지 = 0(메모가 없을 뿐이라 잡음 있음).
지표(48kHz 출력 · librosa · CPU)
  · pesq · stoi · sisdr : SQUIM(d_eval quality 와 같은 모델 · GPU 있으면 GPU)
  · aper   : 말소리 큰 프레임 중 pyin 이 "주기성 낮음"(voiced_prob < 0.5)인 비율 — 거친 결은 주기성이 깨짐
  · jump   : 유성 구간에서 이웃 프레임 F0 가 3반음 넘게 튄 횟수 / 초 — 삑사리(음이 갑자기 꺾임)
  · flat   : 유성 프레임 스펙트럼 평탄도 평균 — 잡음 섞인 결
  · hf     : 유성 프레임 4kHz 이상 에너지 비율
  · 위 넷의 "끝 30%"(_t) — 메모가 대부분 "끝에서 갈라짐"
  ~/miniforge3/envs/voxcpm/bin/python tools/crackle_probe.py test     → ~/workspaces/tmp_listen/vox_koft/pilot/crackle_test.json
함수 feats(path) 는 tools/pilot_bestofn_score.py 가 그대로 씀.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

H = Path.home() / "workspaces"
L6 = H / "tmp_listen/vox_koft/listen_set6"
UR = H / "tmp_listen/user_result"
OUT = H / "tmp_listen/vox_koft/pilot"
KO2 = {"중립": "Neutrality", "기쁨 A(약)": "Happy-A", "기쁨 B(강)": "Happy-B", "슬픔 A(약)": "Sad-A", "슬픔 B(강)": "Sad-B",
       "분노 A(약)": "Angry-A", "분노 B(강)": "Angry-B"}
SR = 16000


def feats(path) -> dict:
    import librosa
    y, _ = librosa.load(str(path), sr=SR)
    hop = 160
    f0, vf, vp = librosa.pyin(y, fmin=60, fmax=900, sr=SR, frame_length=1024, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop)) ** 2
    n = min(len(f0), len(rms), S.shape[1])
    f0, vf, vp, rms, S = f0[:n], vf[:n], vp[:n], rms[:n], S[:, :n]
    loud = rms > 0.1 * rms.max()
    fl = librosa.feature.spectral_flatness(S=S, power=2.0)[0][:n]
    fr = librosa.fft_frequencies(sr=SR, n_fft=1024)
    hf = S[fr >= 4000].sum(0) / (S.sum(0) + 1e-12)
    idx = np.where(loud)[0]

    def part(sel):
        lo = sel & loud
        v = lo & vf
        st = 12 * np.log2(np.where(v, f0, np.nan) / 100.0)
        d = np.abs(np.diff(st))
        jumps = np.nansum((d > 3) & np.isfinite(d))
        sec = max(lo.sum() * hop / SR, 1e-3)
        return dict(aper=float(np.mean(vp[lo] < 0.5)) if lo.any() else np.nan, jump=float(jumps / sec),
                    flat=float(np.mean(fl[v])) if v.any() else np.nan, hf=float(np.mean(hf[v])) if v.any() else np.nan)
    allm = np.ones(n, bool)
    tail = np.zeros(n, bool)
    if len(idx):
        a, b = idx[0], idx[-1]
        tail[int(a + 0.7 * (b - a)): b + 1] = True
    o = part(allm)
    o.update({f"{k}_t": v for k, v in part(tail).items()})
    return o


def squim(paths):
    import soundfile as sf
    import torch
    import torchaudio
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SQUIM_OBJECTIVE.get_model().to(dev).eval()
    out = []
    for p in paths:
        y, sr = sf.read(p, dtype="float32")
        with torch.no_grad():
            z = torchaudio.functional.resample(torch.from_numpy(y), sr, 16000)[None].to(dev)
            out.append(dict(zip(("stoi", "pesq", "sisdr"), (float(v) for v in m(z)))))
    return out


def labels():
    """(rid, tag) → 1/0 — 메모를 손으로 옮긴 규칙(문서 표와 같음)."""
    S = json.load(open(L6 / "set.json"))
    rid_of = {(r["spk"], r["emo"], r["text"]): r["rid"] for r in S["rows"]}
    pages = {"vox_koft_cmp4.csv": ["NUX2sNlr5e5_s3000", "A7NX3_s4500", "X2Rs6lr5e5_s4500", "X2Rs01lr5e5_s4000"],
             "vox_koft_run_NUX2sNplr5e5.csv": ["NUX2sNlr5e5_s3000", "NUX2sNplr5e5_s3500", "NUX2sNplr5e5_s4000", "NUX2sNplr5e5_s4500"]}
    lab = {}
    for f, models in pages.items():
        rows = list(csv.DictReader(open(UR / f, encoding="utf-8-sig")))
        memo = {}
        for r in rows:
            if r["칸"] not in KO2:
                continue
            k = rid_of.get((r["화자"], KO2[r["칸"]], r["문장"]))
            if k and r["메모"].strip():
                memo[k] = r["메모"]
        for (s, e, t), rid in rid_of.items():
            if e not in ("Happy-B", "Sad-B", "Angry-B"):
                continue
            m = memo.get(rid, "")
            # 한 메모에 여러 내용이 섞이면 "갈라짐 · 삑 · 클리핑 · 떨리" 가 든 문장만 봄
            sent = [x for x in m.replace("\n", ". ").split(".") if any(w in x for w in ("갈라", "갈리", "삑", "클리핑", "떨리", "흔들리"))]
            for t_ in models:
                pos = False
                for x in sent:
                    named = [mm for mm in models if mm in x or mm.split("_s")[0] + "_s" + mm.split("_s")[1] in x]
                    s35 = "3500" in x and "4000" in x and "NUX2sNplr5e5" in x
                    if t_ in x or (s35 and t_ in ("NUX2sNplr5e5_s3500", "NUX2sNplr5e5_s4000")):
                        pos = True
                    elif any(w in x for w in ("네 개", "공통적", "전반적")):
                        pos = True
                    elif "비교" in x and ("나머지" in x or "제외" in x):
                        pos = pos or t_ != "NUX2sNlr5e5_s3000"
                    elif not named and not any(mm in x for mm in models):
                        pos = pos or False
                lab[(rid, t_)] = max(lab.get((rid, t_), 0), int(pos))
    return lab


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return np.nan
    return float((np.mean(p[:, None] > n[None]) + 0.5 * np.mean(p[:, None] == n[None])))


def test(a=None):
    lab = labels()
    keys = sorted(lab)
    paths = [str(L6 / "wav" / t / f"{rid}.wav") if (L6 / "wav" / t / f"{rid}.wav").exists()
             else str(H / "tmp_listen/vox_koft/d_eval/wav" / t / f"{rid}.wav") for rid, t in keys]   # 새 줄은 listen_set6 · 나머지는 d_eval 음원
    from multiprocessing import Pool
    with Pool(16) as pool:
        F = pool.map(feats, paths)
    Q = squim(paths)
    y = [lab[k] for k in keys]
    rows = [dict(rid=k[0], tag=k[1], y=lab[k], **f, **q) for k, f, q in zip(keys, F, Q)]
    res = {}
    for m in ["pesq", "stoi", "sisdr", "aper", "jump", "flat", "hf", "aper_t", "jump_t", "flat_t", "hf_t"]:
        sgn = -1 if m in ("pesq", "stoi", "sisdr") else 1                     # 높을수록 "갈라짐 쪽" 으로 맞춤
        res[m] = auc(y, [sgn * r[m] for r in rows])
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(n=len(y), pos=int(sum(y)), auc=res, rows=rows), open(OUT / "crackle_test.json", "w"), ensure_ascii=False, indent=1)
    print(f"표본 {len(y)} · 갈라짐 {sum(y)}")
    for m, v in sorted(res.items(), key=lambda kv: -abs(kv[1] - 0.5)):
        print(f"  {m:8s} AUC {v:.2f}")


if __name__ == "__main__":
    {"test": test}[sys.argv[1]]()
