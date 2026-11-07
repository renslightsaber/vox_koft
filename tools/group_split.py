#!/usr/bin/env python
"""D 재측정을 화자 묶음별로 나눠 보기 — 542 8명(무감정 낭독만 녹음) vs unseen8 8명(감정 녹음 있음) (2026-11-07 · 지시문 q3tts 세션 15:30 §6-4).

왜: X2R02(A7NX2R lr 5e-5)의 감정 ↓ 가 542 화자에 몰려 있다는 q3tts 세션 분해(542 A칸 e2v LoRA01 42% → X2R02 s4500 19%)를
    같은 잣대로 모든 모델에 대해 다시 재고, A7NX2Rs · A7NX2Rs6(542 양만 다름)으로 "542 지름길" 가설을 본다.
값(묶음 × A칸 · B칸, 칸마다 화자 8명 × 감정 3 × 문장 2 = 48개)
  e2v 맞힘 · arousal · 목표 유사도(WavLM · ERes2Net 평균 cos) · 목표 1위(두 잣대 평균) · CER · 이상(d_eval odd)
부록: 화자별 e2v 맞힘(감정 6칸 × 2 = 12개 중)
    ~/miniforge3/envs/voxcpm/bin/python tools/group_split.py --tags base LoRA01_s1500 … [--out docs/reports/figs/x.md]
숫자는 참고용 — 판정은 청음.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voxko.eval import d_eval as DE  # noqa: E402

D = Path.home() / "workspaces/tmp_listen/vox_koft/d_eval"
GROUPS = [("542", "542 8명"), ("unseen8", "unseen8 8명")]


def load(tag):
    s, q = json.load(open(D / f"sv_{tag}.json")), json.load(open(D / f"ser_{tag}.json"))
    qa = json.load(open(D / f"quality_{tag}.json")) if (D / f"quality_{tag}.json").exists() else {}
    return s, q, qa


def stats(P, tag, thr):
    s, q, qa = load(tag)
    out = {}
    for g, _ in GROUPS:
        for ab in ("A", "B"):
            xs = [x for x in P["items"] if P["spk"][x["tgt"]]["src"] == g and x["emo"].endswith("-" + ab)]
            i = [x["id"] for x in xs]
            out[f"{g}|{ab}"] = dict(
                n=len(xs), e2v=100 * np.mean([q[x["id"]]["top"] == DE.E2V[DE.base(x["emo"])] for x in xs]),
                aro=float(np.mean([q[k]["arousal"] for k in i])), sim=float(np.mean([(s[k]["t_sim"] + s[k]["e_t_sim"]) / 2 for k in i])),
                top1=100 * np.mean([(s[k]["top1"] + s[k]["e_top1"]) / 2 for k in i]), cer=100 * np.mean([q[k]["cer"] for k in i]),
                odd=100 * np.mean([bool(DE.odd(qa[k], thr)) for k in i]) if qa else float("nan"))
    per = {}
    for sp in P["spk"]:
        xs = [x for x in P["items"] if x["tgt"] == sp and x["emo"] != "Neutrality"]
        per[sp] = int(sum(q[x["id"]]["top"] == DE.E2V[DE.base(x["emo"])] for x in xs))
    return out, per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    DE.OUT = D
    thr = DE._sq_thr()
    P = json.load(open(D / "plan.json"))
    R = {t: stats(P, t, thr) for t in a.tags if (D / f"ser_{t}.json").exists() and (D / f"sv_{t}.json").exists()}
    md = ["### 핵심 표 — e2v 맞힘 (칸마다 48개 · ±7%p 는 잡음)", "",
          "| 모델 | 542 A | 542 B | unseen8 A | unseen8 B |", "|---|---|---|---|---|"]
    for t, (o, _) in R.items():
        md.append(f"| {t} | " + " | ".join(f"{o[k]['e2v']:.0f}%" for k in ("542|A", "542|B", "unseen8|A", "unseen8|B")) + " |")
    for key, lab, f in (("aro", "arousal", "{:.3f}"), ("sim", "목표 유사도", "{:.3f}"), ("top1", "목표 1위(두 잣대)", "{:.0f}%"),
                        ("cer", "CER", "{:.1f}%"), ("odd", "이상", "{:.0f}%")):
        md += ["", f"### {lab}", "", "| 모델 | 542 A | 542 B | unseen8 A | unseen8 B |", "|---|---|---|---|---|"]
        for t, (o, _) in R.items():
            md.append(f"| {t} | " + " | ".join(f.format(o[k][key]) for k in ("542|A", "542|B", "unseen8|A", "unseen8|B")) + " |")
    spk = list(P["spk"])
    md += ["", "### 부록 — 화자별 e2v 맞힘(감정 6칸 × 2 = 12개 중)", "",
           "| 화자 | 묶음 | " + " | ".join(R) + " |", "|---|---|" + "---|" * len(R)]
    for sp in spk:
        md.append(f"| {sp} | {P['spk'][sp]['src']} | " + " | ".join(str(R[t][1][sp]) for t in R) + " |")
    txt = "\n".join(md)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
        json.dump({t: dict(groups=o, per_speaker=p) for t, (o, p) in R.items()}, open(Path(a.out).with_suffix(".json"), "w"),
                  ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
