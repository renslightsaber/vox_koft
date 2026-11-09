#!/usr/bin/env python
"""모델 간 유사도 조사 — "모델이 달라도 같은 화자는 같은 목소리로 나오는가" (2026-11-09 · 지시문 q3tts 세션 11/08 23:25 §3)

목적: 한 모델로 7칸을 다 잘 만들지 못하면 칸마다 다른 모델로 만들어 합칠 수 있다. 그때 목소리가 이어지는지 미리 본다.
대상: d_eval 통합 점수 상위 5 run × 상위 2 step = 10 ckpt(아래 TAGS). **새 합성 없음** — d_eval 음원(ckpt 마다 224개)을 그대로 씀.
잣대: d_eval sv 와 같은 두 화자 임베딩(WavLM-large · ERes2Net · q3tts_koft q3ko.eval.sv 읽기만) — 결과는 두 잣대를 각각 적음.

  1) emb     (env q3tts · CPU 가능) 임베딩 → cross_model/emb_{w,e}.npz
       CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/q3tts/bin/python tools/cross_model.py emb
  2) analyze (아무 env · numpy/matplotlib) 행렬 · 섞기 모의실험 · 칸 × 모델 표 · 튀는 조합 → cross_model/*.json|csv · docs/reports/figs/
       ~/miniforge3/envs/voxcpm/bin/python tools/cross_model.py analyze
  3) page    작은 청음 페이지(화자 6명 × 한 모델 세트 / 섞은 세트 = 12줄 · d_eval 음원 그대로 · 새 합성 없음)
       ~/miniforge3/envs/voxcpm/bin/python tools/cross_model.py page    # → cross_model/index.html

정의
  - 같은 문항 다른 모델: 같은 (화자, 문장) 의 모델 X 출력 · 모델 Y 출력 코사인 → 224 문항 평균 → 10×10
  - 세트 안 일관성: 화자 1명의 7칸 세트(칸마다 2문장 = 14개)에서 **다른 칸끼리** 모든 쌍의 평균 코사인
    (진짜 녹음 세트는 unseen8 화자 칸마다 1개 = 7개라 같은 칸 쌍이 없으므로, 비교를 맞추려고 모델 세트도 다른 칸 쌍만 셈)
  - 섞기: (가) 한 모델로 7칸 · (나) 칸마다 그 칸 e2v 맞힘이 가장 좋은 모델 · (다) 칸마다 목표 유사도가 가장 높은 모델
    고르기는 d_eval summary 의 칸별(16명 전체) 숫자로 · 같으면 통합 점수가 높은 모델
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

H = Path.home() / "workspaces"
D = H / "tmp_listen/vox_koft/d_eval"
OUT = H / "tmp_listen/vox_koft/cross_model"
FIG = Path(__file__).resolve().parents[1] / "docs/reports/figs"
TAGS = ["NUX2sNlr5e5_s3000", "NUX2sNlr5e5_s4500", "X2Rs6lr5e5_s4500", "X2Rs6lr5e5_s2500",
        "NUX2slr5e5_s4500", "NUX2slr5e5_s4000", "A7NX3_s4500", "A7NX3_s2250",
        "X2Rs01lr5e5_s4000", "X2Rs01lr5e5_s2500"]
EMO7 = ["Neutrality", "Happy-A", "Happy-B", "Sad-A", "Sad-B", "Angry-A", "Angry-B"]
KEYS = {"w": "WavLM", "e": "ERes2Net"}


def run_of(t: str) -> str:
    return t.rsplit("_s", 1)[0]


def check_tags():
    """TAGS 가 지금도 상위 5 run × 상위 2 step 인지 summary 로 다시 확인."""
    S = json.load(open(D / "summary.json"))["models"]
    runs = {}
    for t, v in S.items():
        if "_s" in t and t.rsplit("_s", 1)[1].isdigit():
            runs.setdefault(run_of(t), []).append((v["전체"]["score"], t))
    top = sorted(runs.items(), key=lambda kv: -max(kv[1])[0])[:5]
    want = [t for _, v in top for _, t in sorted(v, reverse=True)[:2]]
    if want != TAGS:
        print("⚠️ 지금 summary 기준 상위 10:", want)
    return want == TAGS


# ───────────────────────────── 1) 임베딩 (env q3tts)
def cmd_emb(a):
    import librosa
    import torch
    torch.set_num_threads(int(a.threads))
    sys.path.insert(0, str(H / "q3tts_koft"))
    from q3ko.eval.sv import load_embedder
    P = json.load(open(D / "plan.json"))
    R = json.load(open(D / "real_cells.json"))["cells"]
    paths = {}                                                       # key → wav
    for t in TAGS:
        for x in P["items"]:
            paths[f"{t}|{x['id']}"] = D / "wav" / t / f"{x['id']}.wav"
    for k, v in R.items():                                           # unseen8 진짜 사람 칸 녹음(없는 칸 = None 은 뺌)
        if not v:
            continue
        paths[f"real|{k}"] = D / v["file"]
    for s, v in P["spk"].items():                                    # 목표 화자 진짜 무감정 녹음(d_eval 목표 평균에 쓰는 것)
        for j, p in enumerate(v["real"]):
            paths[f"spkreal|{s}|{j}"] = Path(p)
    miss = [k for k, p in paths.items() if not Path(p).is_file()]
    if miss:
        raise SystemExit(f"없는 음원 {len(miss)}개 — 예: {miss[:3]}")
    dev = "cuda:0" if (a.device == "cuda") else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    for k, name in (("w", "wavlm_large"), ("e", "eres2net")):
        f = OUT / f"emb_{k}.npz"
        old = dict(np.load(f)) if f.exists() else {}
        todo = [x for x in paths if x not in old]
        print(f"[{KEYS[k]}] {len(old)} 있음 · {len(todo)} 새로 · {dev}", flush=True)
        if not todo:
            continue
        F = load_embedder(name, device=dev)
        for i, key in enumerate(todo):
            v = F(librosa.load(str(paths[key]), sr=16000)[0], 16000)
            old[key] = np.asarray(v, dtype=np.float32)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
                np.savez(f, **old)
        np.savez(f, **old)
    print("완료", OUT)


# ───────────────────────────── 2) 분석
def unit(v):
    return v / np.linalg.norm(v)


def cmd_analyze(a):
    P = json.load(open(D / "plan.json"))
    S = json.load(open(D / "summary.json"))["models"]
    R = json.load(open(D / "real_cells.json"))["cells"]
    items = P["items"]
    grp = lambda s: "542" if P["spk"][s]["src"] == "542" else "unseen8"   # noqa: E731
    spks = sorted(P["spk"])
    E = {k: {kk: unit(v) for kk, v in np.load(OUT / f"emb_{k}.npz").items()} for k in KEYS}
    sv = {t: json.load(open(D / f"sv_{t}.json")) for t in TAGS}
    res = {"tags": TAGS, "tags_ok": check_tags(), "n_items": len(items)}

    # ① 같은 문항 다른 모델 — 10×10 (전체 · 칸별 · 묶음별)
    def mat(k, flt):
        xs = [x for x in items if flt(x)]
        M = np.zeros((len(TAGS), len(TAGS)))
        for i, ti in enumerate(TAGS):
            for j, tj in enumerate(TAGS):
                M[i, j] = np.mean([E[k][f"{ti}|{x['id']}"] @ E[k][f"{tj}|{x['id']}"] for x in xs])
        return M
    off = ~np.eye(len(TAGS), dtype=bool)
    same_run = np.array([[run_of(a_) == run_of(b_) for b_ in TAGS] for a_ in TAGS]) & off
    diff_run = off & ~same_run
    res["matrix"], res["matrix_summary"] = {}, {}
    for k in KEYS:
        res["matrix"][k] = {"전체": mat(k, lambda x: True).round(4).tolist()}
        for e in EMO7:
            res["matrix"][k][e] = mat(k, lambda x, e=e: x["emo"] == e).round(4).tolist()
        for g in ("542", "unseen8"):
            res["matrix"][k][g] = mat(k, lambda x, g=g: grp(x["tgt"]) == g).round(4).tolist()
        summ = {}
        for part, M in res["matrix"][k].items():
            M = np.array(M)
            summ[part] = {"same_run": float(M[same_run].mean()), "diff_run": float(M[diff_run].mean()),
                          "diff_run_min": float(M[diff_run].min())}
        res["matrix_summary"][k] = summ

    # 기준선 ⓪ 같은 모델 · 같은 화자 · 같은 칸 · 다른 문장(s0 vs s1)
    # 기준선 ② 진짜 사람: unseen8 화자 진짜 칸 녹음끼리(다른 칸 쌍) · 542 는 진짜 무감정끼리
    base = {}
    for k in KEYS:
        same_cell = []
        for t in TAGS:
            for s in spks:
                for e in EMO7:
                    ids = [x["id"] for x in items if x["tgt"] == s and x["emo"] == e]
                    for p_, q_ in itertools.combinations(ids, 2):
                        same_cell.append(E[k][f"{t}|{p_}"] @ E[k][f"{t}|{q_}"])
        real_cross = []
        for s in {kk.split("|")[0] for kk in R}:
            v = [E[k][f"real|{s}|{e}"] for e in EMO7 if f"real|{s}|{e}" in E[k]]
            real_cross += [p_ @ q_ for p_, q_ in itertools.combinations(v, 2)]
        neu = []
        for s in spks:
            v = [E[k][kk] for kk in E[k] if kk.startswith(f"spkreal|{s}|")]
            neu += [p_ @ q_ for p_, q_ in itertools.combinations(v, 2)]
        base[k] = {"same_model_same_cell_diff_sent": float(np.mean(same_cell)),
                   "real_unseen8_diff_cells": float(np.mean(real_cross)), "real_unseen8_n_pairs": len(real_cross),
                   "real_neutral_pairs_all16": float(np.mean(neu)) if neu else None}
    res["baseline"] = base
    res["per_model_cross"] = {k: {t: float(np.mean([np.array(res["matrix"][k]["전체"])[i, j] for j, u in enumerate(TAGS) if run_of(u) != run_of(t)]))
                                  for i, t in enumerate(TAGS)} for k in KEYS}

    # ② 칸 섞어 만들기 모의실험
    cell = {t: S[t]["칸별"] for t in TAGS}
    score = {t: S[t]["전체"]["score"] for t in TAGS}
    pick = {"(나) e2v 최고": {e: max(TAGS, key=lambda t: (cell[t][e]["emo_hit"], score[t])) for e in EMO7},
            "(다) 목표 유사도 최고": {e: max(TAGS, key=lambda t: (cell[t][e]["t_sim"], score[t])) for e in EMO7}}
    # (라) = (나) 를 같은 계열(lr 5e-5 · 다른 칸 ref 2개 계열 8개)에서만 — 다른 모델과 가장 멀었던 A7NX3(lr 1e-4 · ref 3개)를 빼면?
    fam = [t for t in TAGS if not t.startswith("A7NX3")]
    pick["(라) e2v 최고 · A7NX3 빼고"] = {e: max(fam, key=lambda t: (cell[t][e]["emo_hit"], score[t])) for e in EMO7}

    def set_items(s, choice):                                       # choice: 칸 → tag
        return [(choice[x["emo"]], x) for x in items if x["tgt"] == s]

    def consistency(k, s, choice, detail=None):
        its = set_items(s, choice)
        vals = []
        for (ta, xa), (tb, xb) in itertools.combinations(its, 2):
            if xa["emo"] == xb["emo"]:
                continue
            c_ = float(E[k][f"{ta}|{xa['id']}"] @ E[k][f"{tb}|{xb['id']}"])
            vals.append(c_)
            if detail is not None and ta != tb:
                detail.append((s, xa, xb, ta, tb, c_))
        return float(np.mean(vals))

    def tsim(k, s, choice):
        f = "t_sim" if k == "w" else "e_t_sim"
        return float(np.mean([sv[t][x["id"]][f] for t, x in set_items(s, choice)]))

    real_spk = sorted({kk.split("|")[0] for kk in R})
    mix = {}
    for k in KEYS:
        rows = {}
        for t in TAGS:
            ch = {e: t for e in EMO7}
            rows[f"(가) {t}"] = {s: (consistency(k, s, ch), tsim(k, s, ch)) for s in spks}
        for name, ch in pick.items():
            rows[name] = {s: (consistency(k, s, ch), tsim(k, s, ch)) for s in spks}
        out = {}
        for name, per in rows.items():
            out[name] = {g: {"cons": float(np.mean([per[s][0] for s in spks if g == "전체" or grp(s) == g])),
                             "tsim": float(np.mean([per[s][1] for s in spks if g == "전체" or grp(s) == g]))}
                         for g in ("전체", "542", "unseen8")}
        single = [out[f"(가) {t}"] for t in TAGS]
        out["(가) 최고"] = {g: {"cons": max(x[g]["cons"] for x in single)} for g in ("전체", "542", "unseen8")}
        out["(가) 평균"] = {g: {"cons": float(np.mean([x[g]["cons"] for x in single]))} for g in ("전체", "542", "unseen8")}
        rv = []
        for s in real_spk:
            v = [E[k][f"real|{s}|{e}"] for e in EMO7 if f"real|{s}|{e}" in E[k]]
            rv.append(np.mean([p_ @ q_ for p_, q_ in itertools.combinations(v, 2)]))
        out["진짜 녹음(unseen8 · 칸마다 1개)"] = {"unseen8": {"cons": float(np.mean(rv))}}
        mix[k] = out
    res["pick"], res["mix"] = pick, mix
    res["pick_cells"] = {n: {"e2v": float(np.mean([cell[ch[e]][e]["emo_hit"] for e in EMO7])),
                             "t_sim": float(np.mean([cell[ch[e]][e]["t_sim"] for e in EMO7]))} for n, ch in
                         list(pick.items()) + [(f"(가) {t}", {e: t for e in EMO7}) for t in TAGS]}

    # 같은 단일 모델 세트 쌍과 견준 "튀는 조합" — (나) 섞기에서 다른 모델끼리인 (화자, 칸 쌍)
    # 낙차 = (모델 A 만으로 같은 두 문항 · 모델 B 만으로 같은 두 문항 평균) − 섞은 쌍
    worst = {}
    for k in KEYS:
        lst = []
        for name, ch in pick.items():
            det = []
            for s in spks:
                consistency(k, s, ch, det)
            for s, xa, xb, ta, tb, c_ in det:
                ref_ = (E[k][f"{ta}|{xa['id']}"] @ E[k][f"{ta}|{xb['id']}"] + E[k][f"{tb}|{xa['id']}"] @ E[k][f"{tb}|{xb['id']}"]) / 2
                lst.append({"mix": name, "spk": s, "group": grp(s), "a": xa["id"], "b": xb["id"], "model_a": ta, "model_b": tb,
                            "cos": round(c_, 4), "same_model_cos": round(float(ref_), 4), "drop": round(float(ref_ - c_), 4)})
        # 같은 (화자 · 모델 쌍 · 칸 쌍) 은 문장 2×2 를 평균해서 하나로
        agg = {}
        for r in lst:
            ea, eb = r["a"].rsplit("_", 2)[-2], r["b"].rsplit("_", 2)[-2]
            key = (r["mix"], r["spk"], r["model_a"], ea, r["model_b"], eb)
            agg.setdefault(key, []).append(r)
        rows = []
        for key, rs in agg.items():
            rows.append({"mix": key[0], "spk": key[1], "group": rs[0]["group"], "model_a": key[2], "cell_a": key[3],
                         "model_b": key[4], "cell_b": key[5], "cos": round(float(np.mean([r["cos"] for r in rs])), 4),
                         "same_model_cos": round(float(np.mean([r["same_model_cos"] for r in rs])), 4),
                         "drop": round(float(np.mean([r["drop"] for r in rs])), 4), "pairs": [(r["a"], r["b"]) for r in rs]})
        worst[k] = sorted(rows, key=lambda r: -r["drop"])[:10]
        worst[k + "_low"] = sorted(rows, key=lambda r: r["cos"])[:10]
    res["worst"] = worst

    # ③ 칸 × 모델 표
    res["cell_table"] = {t: {e: {m: round(float(cell[t][e][m]), 4) for m in ("emo_hit", "t_sim", "top1", "e_top1", "cer")} for e in EMO7}
                         for t in TAGS}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "cross_model.json", "w"), ensure_ascii=False, indent=1)
    with open(OUT / "matrix_전체.csv", "w", newline="") as f:
        w = csv.writer(f)
        for k in KEYS:
            w.writerow([KEYS[k]] + TAGS)
            for t, row in zip(TAGS, res["matrix"][k]["전체"]):
                w.writerow([t] + row)
            w.writerow([])
    with open(OUT / "cell_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "cell", "emo_hit", "t_sim", "top1", "e_top1", "cer"])
        for t in TAGS:
            for e in EMO7:
                c_ = res["cell_table"][t][e]
                w.writerow([t, e, c_["emo_hit"], c_["t_sim"], c_["top1"], c_["e_top1"], c_["cer"]])
    with open(OUT / "worst_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["embedder", "mix", "spk", "group", "model_a", "cell_a", "model_b", "cell_b", "cos", "same_model_cos", "drop"])
        for k in KEYS:
            for r in worst[k]:
                w.writerow([KEYS[k], r["mix"], r["spk"], r["group"], r["model_a"], r["cell_a"], r["model_b"], r["cell_b"],
                            r["cos"], r["same_model_cos"], r["drop"]])
    figs(res)
    report_print(res)


def figs(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",):
        if os.path.isfile(fp):
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    short = [t.replace("lr5e5", "") for t in TAGS]
    # 그림 1: 10×10 heatmap(두 잣대) — 한 색(파랑) 밝음 → 어두움
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, k in zip(axs, KEYS):
        M = np.array(res["matrix"][k]["전체"])
        vals = M[~np.eye(len(TAGS), dtype=bool)]
        im = ax.imshow(M, cmap="Blues", vmin=vals.min() - 0.02, vmax=1.0)
        for i in range(len(TAGS)):
            for j in range(len(TAGS)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > (vals.min() + 1) / 2 else "#222")
        ax.set_xticks(range(len(TAGS)), short, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(TAGS)), short, fontsize=8)
        for b in range(0, len(TAGS), 2):                              # 같은 run 두 step 을 상자로
            ax.add_patch(plt.Rectangle((b - 0.5, b - 0.5), 2, 2, fill=False, ec="#d62728", lw=1.5))
        s = res["matrix_summary"][k]["전체"]
        ax.set_title(f"{KEYS[k]} - same item, model X vs model Y (cosine)\nsame run other step {s['same_run']:.3f} / other run {s['diff_run']:.3f}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Cross-model voice similarity (D re-eval, mean of 224 items; red box = same run)", fontsize=12)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "20261109_cross_model_matrix.png", dpi=130)
    plt.close(fig)
    # 그림 2: 섞기 비교 막대(세트 안 일관성 · 두 잣대 · 전체/542/unseen8)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), sharey=False)
    for ax, k in zip(axs, KEYS):
        mx = res["mix"][k]
        names = [f"(가) {t}" for t in TAGS] + ["(나) e2v 최고", "(라) e2v 최고 · A7NX3 빼고", "(다) 목표 유사도 최고"]
        en = {"(나) e2v 최고": "MIX best e2v per cell", "(라) e2v 최고 · A7NX3 빼고": "MIX best e2v, no A7NX3", "(다) 목표 유사도 최고": "MIX best voice per cell"}
        y = [mx[n]["전체"]["cons"] for n in names]
        col = ["#9ecae1"] * len(TAGS) + ["#e6550d", "#fd8d3c", "#31a354"]
        ax.barh(range(len(names)), y, color=col, height=0.7)
        for i, v in enumerate(y):
            ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
        rv = mx["진짜 녹음(unseen8 · 칸마다 1개)"]["unseen8"]["cons"]
        ax.axvline(rv, color="#555", ls="--", lw=1)
        ax.text(rv, len(names) - 0.3, f" real recordings\n (unseen8) {rv:.3f}", fontsize=8, color="#555", va="bottom")
        ax.set_yticks(range(len(names)), [en.get(n, n.replace("(가) ", "single ").replace("lr5e5", "")) for n in names], fontsize=8)
        ax.invert_yaxis()
        lo = min(min(y), rv) - 0.03
        ax.set_xlim(lo, max(max(y), rv) + 0.03)
        ax.set_xlabel("set consistency (mean cosine over cross-cell pairs, 16 speakers)")
        ax.set_title(KEYS[k], fontsize=10)
    fig.suptitle("One model for all 7 cells vs a different model per cell", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "20261109_cross_model_mix.png", dpi=130)
    plt.close(fig)


def report_print(res):
    for k in KEYS:
        s, b = res["matrix_summary"][k], res["baseline"][k]
        print(f"== {KEYS[k]}")
        print(f"  같은 문항: 같은 run 다른 step {s['전체']['same_run']:.3f} · 다른 run {s['전체']['diff_run']:.3f} (최저 {s['전체']['diff_run_min']:.3f})"
              f" · 같은 모델 같은 칸 다른 문장 {b['same_model_same_cell_diff_sent']:.3f} · 진짜 unseen8 다른 칸 {b['real_unseen8_diff_cells']:.3f}"
              f" · 진짜 무감정끼리 {b['real_neutral_pairs_all16']:.3f}")
        for part in EMO7 + ["542", "unseen8"]:
            print(f"   {part:10s} 같은 run {s[part]['same_run']:.3f} · 다른 run {s[part]['diff_run']:.3f} (최저 {s[part]['diff_run_min']:.3f})")
        for n, v in res["mix"][k].items():
            print(f"  {n:34s} " + " · ".join(f"{g} {x['cons']:.3f}" + (f"/t{x['tsim']:.3f}" if "tsim" in x else "") for g, x in v.items()))
    for n, ch in res["pick"].items():
        print(n, {e: t for e, t in ch.items()})
    for n, v in res["pick_cells"].items():
        print(f"  {n:34s} 7칸 e2v {v['e2v']:.1f} · 목표 유사도 {v['t_sim']:.3f}")
    for k in KEYS:
        print(KEYS[k], "모델마다 다른 run 과 평균:", {t: round(v, 3) for t, v in res["per_model_cross"][k].items()})


# ───────────────────────────── 3) 작은 청음 페이지
SINGLE = "NUX2sNlr5e5_s3000"                   # 한 모델 세트(통합 1위)
MIX = "(라) e2v 최고 · A7NX3 빼고"             # 섞은 세트(권장 조합)
PAGE_JS = """
const K='cross_judg';let J=JSON.parse(localStorage.getItem(K)||'{}');
function save(){localStorage.setItem(K,JSON.stringify(J));document.getElementById('cnt').textContent=Object.values(J).filter(v=>v.v).length;}
document.querySelectorAll('.b').forEach(d=>{const id=d.dataset.id;d.querySelectorAll('button').forEach(b=>{if((J[id]||{}).v===b.dataset.v)b.classList.add('on');
 b.onclick=()=>{J[id]=J[id]||{};J[id].v=(J[id].v===b.dataset.v)?'':b.dataset.v;d.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.v===J[id].v));save();};});});
document.querySelectorAll('textarea[data-id]').forEach(t=>{const id=t.dataset.id;t.value=(J[id]||{}).memo||'';t.oninput=()=>{J[id]=J[id]||{};J[id].memo=t.value;save();};});
function q(s){s=String(s==null?'':s);return /[",\\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}
function dl(){const rows=[['row','speaker','set','judgement','memo']];document.querySelectorAll('tr[data-id]').forEach(r=>{const id=r.dataset.id,j=J[id]||{};rows.push([id,r.dataset.spk,r.dataset.set,j.v||'',j.memo||'']);});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+rows.map(r=>r.map(q).join(',')).join('\\n')],{type:'text/csv'}));a.download='vox_koft_cross_model.csv';a.click();}
save();
"""


def cmd_page(a):
    import html
    E = html.escape
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import listen_pages as LP
    P = json.load(open(D / "plan.json"))
    R = json.load(open(OUT / "cross_model.json"))
    mix = R["pick"][MIX]
    ko = {"Neutrality": "중립", "Happy-A": "기쁨A", "Happy-B": "기쁨B", "Sad-A": "슬픔A", "Sad-B": "슬픔B", "Angry-A": "분노A", "Angry-B": "분노B"}
    sh = lambda t: t.replace("lr5e5", "")  # noqa: E731
    A = lambda p: f'<audio controls preload="none" src="{p}"></audio>'  # noqa: E731
    miss = []
    h = [f"<!doctype html><meta charset=utf-8><title>모델 섞기 청음</title><style>{LP.CSS}"
         ".b button.on{background:#333;color:#fff} td small{color:#666}</style>",
         "<div id=bar>판정 <span id=cnt>0</span>/12 · <button onclick='dl()'>CSV 내려받기</button></div>",
         "<h1>🎧 vox_koft · 모델 섞기 청음 — 한 모델 세트 vs 칸마다 다른 모델</h1>",
         "<ul><li>화자 6명 × 2줄 = 12줄. 줄 하나 = 그 화자의 7칸(같은 REF · 칸마다 문장 1개 · D 재측정 음원 그대로).</li>"
         f"<li><b>한 모델</b> = {E(SINGLE)}(통합 1위) · <b>섞음</b> = 칸마다 그 칸 e2v 맞힘이 가장 좋은 모델(A7NX3 은 다른 모델과 목소리가 가장 멀어 뺌 — 보고서 §3).</li>"
         "<li>물음: <b>한 줄의 7칸이 모두 REF 와 같은 사람으로 들리는가</b>(같은 사람 · 애매 · 다름). 감정 세기는 보지 않아도 됩니다.</li>"
         "<li>판정 · 메모는 이 브라우저에 저장되고 CSV 로 받습니다(저장 키 cross_judg).</li></ul>",
         "<table><tr><th>화자</th><th>세트</th><th>REF</th>" + "".join(f"<th>{ko[e]}</th>" for e in EMO7) + "<th>판정</th><th>메모</th></tr>"]
    for s in LP.SPK6:
        for name, ch in (("한 모델", {e: SINGLE for e in EMO7}), ("섞음", mix)):
            rid = f"{s}|{'single' if name == '한 모델' else 'mix'}"
            cells = []
            for e in EMO7:
                x = next(y for y in P["items"] if y["tgt"] == s and y["emo"] == e and y["id"].endswith("_0"))
                f = D / "wav" / ch[e] / f"{x['id']}.wav"
                miss += [] if f.exists() else [str(f)]
                xid = x["id"]
                cells.append(f"<td>{A(f'../d_eval/wav/{ch[e]}/{xid}.wav')}<br><small>{E(sh(ch[e]))}</small></td>")
            refp = D / "ref" / f"{s}_ref.wav"
            miss += [] if refp.exists() else [str(refp)]
            btn = "<div class=b data-id='%s'>%s</div>" % (E(rid), "".join(f"<button data-v='{o}'>{o}</button>" for o in ("같은 사람", "애매", "다름")))
            h.append(f"<tr data-id='{E(rid)}' data-spk='{E(s)}' data-set='{name}'><td>{E(s)}<br><small>{P['spk'][s]['src']} · {P['spk'][s]['gender']}</small></td>"
                     f"<td><b>{name}</b></td><td>{A(f'../d_eval/ref/{s}_ref.wav')}</td>{''.join(cells)}<td>{btn}</td>"
                     f"<td><textarea data-id='{E(rid)}' rows=2 cols=18></textarea></td></tr>")
    h.append("</table><p class=n>숫자 근거: docs/reports/20261109_cross_model_similarity.md · 새 합성 없음(D 재측정 음원은 이미 재현 확인을 거친 것).</p>")
    h.append(f"<script>{PAGE_JS}</script>")
    if miss:
        raise SystemExit(f"없는 음원 {len(miss)}: {miss[:3]}")
    (OUT / "index.html").write_text("\n".join(h), encoding="utf-8")
    print("✅", OUT / "index.html", "· 12줄")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("emb")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--threads", default=32)
    sub.add_parser("analyze")
    sub.add_parser("page")
    a = ap.parse_args()
    {"emb": cmd_emb, "analyze": cmd_analyze, "page": cmd_page}[a.cmd](a)
