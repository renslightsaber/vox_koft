#!/usr/bin/env python
"""칸별 "생성 중립 대비" 목소리 이동 표 (2026-11-07) — docs/reports/20261106_cell_voice_drift_analysis.md §3 와 같은 잣대.

값 = 같은 모델 · 같은 목표 화자의 **생성 중립 2개 평균** ↔ 각 감정 칸 출력(문장 2개)의 cos 유사도 · WavLM-large · ERes2Net 평균 · 목표 16명 평균
재료 = D 재측정 임베딩 tmp_listen/vox_koft/d_eval/vspk/emb_<tag>.npz(tools/vspk.py emb) · 진짜 녹음 천장 = vspk emb_real(같은 구조)

    python tools/cell_drift.py --tags base LoRA01_s1500 A7NX_s1500 A7NX3_s750 … [--out <md>]
칸마다 32개라 ±0.01~0.02 는 잡음 수준. 숫자는 참고용 — 판정은 청음.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path.home() / "workspaces/tmp_listen/vox_koft/d_eval/vspk"
CELLS = ["Happy-A", "Happy-B", "Sad-A", "Sad-B", "Angry-A", "Angry-B"]
KO = {"Happy-A": "기쁨A", "Happy-B": "기쁨B", "Sad-A": "슬픔A", "Sad-B": "슬픔B", "Angry-A": "분노A", "Angry-B": "분노B"}


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cells(tag: str) -> dict:
    z = np.load(D / f"emb_{tag}.npz")
    ids = [str(i) for i in z["ids"]]
    out = defaultdict(list)
    for k in ("w", "e"):
        nb = defaultdict(list)
        for i, v in zip(ids, z[k]):
            if "_Neutrality_" in i:
                nb[i.split("_Neutrality_")[0]].append(unit(v))
        nc = {s: unit(np.mean(v, 0)) for s, v in nb.items()}
        for i, v in zip(ids, z[k]):
            if "_Neutrality_" in i:
                continue
            s, e, _ = i.rsplit("_", 2)
            out[(e, i)].append(float(unit(v) @ nc[s]))
    by = defaultdict(list)
    for (e, _), v in out.items():
        by[e].append(np.mean(v))
    r = {e: float(np.mean(by[e])) for e in CELLS}
    r["B평균"] = float(np.mean([r[e] for e in CELLS if e.endswith("B")]))
    r["A평균"] = float(np.mean([r[e] for e in CELLS if e.endswith("A")]))
    return r


def real() -> dict:
    z = np.load(D / "emb_real.npz")
    meta = json.loads(str(z["meta"]))
    sims = defaultdict(list)
    for s in sorted({m["spk"] for m in meta}):
        idx = [j for j, m in enumerate(meta) if m["spk"] == s]
        nidx = [j for j in idx if meta[j]["grp"] == "Neutrality"]
        for j in idx:
            g = meta[j]["grp"]
            if g in ("Happy", "Sad", "Angry"):
                sims[g].append(float(np.mean([unit(z[k][j]) @ unit(np.mean([unit(z[k][n]) for n in nidx], 0)) for k in ("w", "e")])))
    return {g: float(np.mean(v)) for g, v in sims.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = {t: cells(t) for t in a.tags if (D / f"emb_{t}.npz").exists()}
    miss = [t for t in a.tags if t not in R]
    rr = real()
    cols = CELLS + ["A평균", "B평균"]
    md = ["| 모델 | " + " | ".join(KO.get(c, c) for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for t, r in R.items():
        md.append(f"| {t} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    md.append("| 진짜 녹음 | " + " | ".join(f"{rr[c.split('-')[0]]:.3f}" if "-" in c else "-" for c in cols) + " |")
    if miss:
        md.append(f"\n(임베딩 없음: {miss})")
    txt = "\n".join(md)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
        json.dump(dict(models=R, real=rr), open(Path(a.out).with_suffix(".json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
