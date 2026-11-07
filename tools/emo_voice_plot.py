#!/usr/bin/env python
"""감정 ↔ 목소리 그림 (2026-11-07 · 지시문 q3tts 세션 11:45 §5-1) — "맞바꿈 곡선을 넘었는가"를 한눈에.

점 = D 재측정(목표 16명 × 감정 6칸 × 문장 2 = 192개) 한 모델(run · step)
  x = e2v 감정 맞힘(6칸) · y = 목표 화자 유사도(6칸 · WavLM-large 와 ERes2Net 평균 cos)
  run 마다 step 순서로 선 · 원본(★) · CFG 시험 점(◆, tools/cfg_test.py) · 오차 막대 = ±1 표준오차(e2v 약 ±3.6%p · 유사도는 192개 표준오차)
재료: ~/workspaces/tmp_listen/vox_koft/d_eval/{sv,ser}_<tag>.json · cfg_test/{sv,ser}_<tag>.json
    ~/miniforge3/envs/voxcpm/bin/python tools/emo_voice_plot.py [--out docs/reports/figs/20261107_emo_voice.png]
숫자는 참고용 — 판정은 청음.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voxko.eval import d_eval as DE  # noqa: E402

H = Path.home() / "workspaces/tmp_listen/vox_koft"
D, C = H / "d_eval", H / "cfg_test"
# 고정 순서(색 = run, 순위가 아니라 항목을 따름) — 기본 팔레트 1~6번
SERIES = [("LoRA01", "LoRA_01 (A7N, neutral ref)", "#2a78d6"), ("A7NX", "A7NX (other-cell ref x1)", "#eb6834"),
          ("A7NX3", "A7NX3 (ref x3)", "#1baf7a"), ("X2R01", "A7NX2R lr 1e-4", "#eda100"),
          ("X2R02lr5e5", "A7NX2R lr 5e-5", "#e87ba4"), ("X2Rs01lr5e5", "A7NX2Rs lr 5e-5 (542 5.8%)", "#008300"),
          ("X2Rs6lr5e5", "A7NX2Rs6 lr 5e-5 (542 3.0%)", "#6250d6")]
CELLS6 = {"Happy-A", "Happy-B", "Sad-A", "Sad-B", "Angry-A", "Angry-B"}


GROUP = None                                    # None = 16명 전체 · '542' · 'unseen8'(--group)


def point(o: Path, tag: str):
    fs, fq = o / f"sv_{tag}.json", o / f"ser_{tag}.json"
    if not (fs.exists() and fq.exists()):
        return None
    P = json.load(open(o / "plan.json"))
    s, q = json.load(open(fs)), json.load(open(fq))
    xs = [x for x in P["items"] if x["emo"] in CELLS6 and x["id"] in s and x["id"] in q
          and (GROUP is None or P["spk"][x["tgt"]]["src"] == GROUP)]
    hit = np.array([q[x["id"]]["top"] == DE.E2V[DE.base(x["emo"])] for x in xs], float)
    sim = np.array([(s[x["id"]]["t_sim"] + s[x["id"]]["e_t_sim"]) / 2 for x in xs])
    return dict(tag=tag, n=len(xs), e2v=100 * hit.mean(), e2v_se=100 * np.sqrt(hit.mean() * (1 - hit.mean()) / len(xs)),
                sim=float(sim.mean()), sim_se=float(sim.std(ddof=1) / np.sqrt(len(xs))))


def collect():
    pts = {"base": point(D, "base")}
    for short, _, _ in SERIES:
        tags = [p.name for p in (D / "wav").iterdir() if re.fullmatch(rf"{short}_s\d+", p.name)]
        ps = [point(D, t) for t in sorted(tags, key=lambda t: int(t.rsplit("_s", 1)[1]))]
        pts[short] = [p for p in ps if p]
    pts["cfg"] = [p for p in (point(C, t.name) for t in sorted((C / "wav").iterdir())) if p] if (C / "wav").exists() else []
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs/reports/figs/20261107_emo_voice.png"))
    ap.add_argument("--group", choices=["542", "unseen8"], help="화자 묶음만(8명 · 점마다 96개)")
    ap.add_argument("--small", action="store_true", help="작은 그림")
    a = ap.parse_args()
    global GROUP
    GROUP = a.group
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    pts = collect()
    fig, ax = plt.subplots(figsize=(7, 4.8) if a.small else (10, 6.5), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#e6e5e0", lw=0.8)
    rows = []
    for short, label, col in SERIES:
        ps = pts.get(short) or []
        if not ps:
            continue
        x, y = [p["e2v"] for p in ps], [p["sim"] for p in ps]
        ax.plot(x, y, "-", color=col, lw=2, alpha=0.85, label=label, zorder=2)
        ax.errorbar(x, y, xerr=[p["e2v_se"] for p in ps], yerr=[p["sim_se"] for p in ps], fmt="o", ms=7, color=col,
                    mec="#fcfcfb", mew=2, elinewidth=0.8, alpha=0.9, zorder=3)
        last = ps[-1]
        ax.annotate(f"s{last['tag'].rsplit('_s', 1)[1]}", (last["e2v"], last["sim"]), xytext=(5, 4), textcoords="offset points",
                    fontsize=8, color="#52514e")
        rows += [(label, p) for p in ps]
    b = pts["base"]
    if b:
        ax.errorbar([b["e2v"]], [b["sim"]], xerr=[b["e2v_se"]], yerr=[b["sim_se"]], fmt="*", ms=16, color="#0b0b0b", mec="#fcfcfb",
                    elinewidth=0.8, label="base VoxCPM2", zorder=4)
        rows.append(("원본", b))
    for p in pts["cfg"]:
        st, cf = re.match(r"X2R02_s(\d+)_cfg(\d+)", p["tag"]).groups()
        ax.errorbar([p["e2v"]], [p["sim"]], xerr=[p["e2v_se"]], yerr=[p["sim_se"]], fmt="D", ms=7, color="#e87ba4", mec="#0b0b0b",
                    mew=1, elinewidth=0.8, zorder=4)
        ax.annotate(f"s{st} cfg{int(cf) / 10:.1f}", (p["e2v"], p["sim"]), xytext=(5, -10), textcoords="offset points", fontsize=8, color="#52514e")
        rows.append(("CFG 시험", p))
    if pts["cfg"]:
        ax.plot([], [], "D", color="#e87ba4", mec="#0b0b0b", label="A7NX2R lr 5e-5 + cfg 2.5 / 3.0 (diamond)")
    ax.set_xlabel("emotion: e2v hit, 6 emotion cells (%)  -> stronger", color="#0b0b0b")
    ax.set_ylabel("voice: target sim  -> closer" if a.small else "voice: target-speaker cos sim (WavLM+ERes2Net mean)  -> closer", color="#0b0b0b")
    ax.set_title((f"{GROUP} speakers (8) only - 96 items per point, +-1 SE" if GROUP else "emotion vs voice - D re-eval, 192 items per point (line = step order, bars = +-1 SE), reference only"), fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=7 if a.small else 8.5, loc="best")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, facecolor=fig.get_facecolor())
    md = ["| 묶음 | 이름표 | e2v 6칸 | 목표 유사도 |", "|---|---|---|---|"] + [f"| {l} | {p['tag']} | {p['e2v']:.1f}% ± {p['e2v_se']:.1f} | {p['sim']:.4f} ± {p['sim_se']:.4f} |" for l, p in rows]
    Path(a.out).with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("✅", a.out)


if __name__ == "__main__":
    main()
