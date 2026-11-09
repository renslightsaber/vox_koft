#!/usr/bin/env python
"""기쁨A 조사 — 행이 가장 많은 칸인데 왜 약한가: 정답 녹음이 실제로 기쁘게 들리는가(이름표 흐림)
(2026-11-09 · 지시문 q3tts 세션 11/09 09:30 §3 · 학습 없음)

대상: A7NUX2sN 학습 정답(utt 중복 제거)
  - 기쁨A: 전부 · 비교 칸 슬픔A · 분노A: 화자마다 최대 N_CMP 개(seed 1986 · 문자열 키)
  - 화자 무감정 기준: tools/angryB_pitch.py 와 같은 무감정 12개(학습 무감정 행 + A7N ref 풀)
잴 것
  - e2v(emotion2vec_plus_large) top 라벨 · happy 확률 · audeering arousal · valence(화자 무감정 평균 대비 차)
  - F0 중앙값(화자 무감정 대비 반음 · angryB_pitch 규칙)
  - 모델 쪽: NUX2sN s3000(D 재측정) 기쁨A 출력의 e2v · 같은 모델 중립 대비 목소리 이동(tools/cell_drift.py 잣대)
판정: 기쁨A 정답 중 "무감정처럼 들리는 녹음"(e2v top = neutral 그리고 happy < 0.2) 비율 ≥ 30% → §5(A7NUX2sNph) 실행

  CUDA_VISIBLE_DEVICES=3 ~/miniforge3/envs/q3tts_ser/bin/python tools/happyA_check.py score     # e2v · audeering (캐시)
  ~/miniforge3/envs/voxcpm/bin/python tools/happyA_check.py f0                                  # CPU
  ~/miniforge3/envs/voxcpm/bin/python tools/happyA_check.py report                              # → happyA_check/summary.json · by_speaker.md
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
H = Path.home() / "workspaces"
V = H / "preprocessed/vox_koft"
TRAIN = V / "A7NUX2sN/train.jsonl"
POOL = V / "A7N/ref_pool.jsonl"
OUT = H / "tmp_listen/vox_koft/happyA_check"
D = H / "tmp_listen/vox_koft/d_eval"
CELLS = ["Happy-A", "Sad-A", "Angry-A"]
N_CMP, N_NEU = 60, 12
F0_UP, MIN_VOICED = 20, 0.3
NEU_LIKE_P = 0.2
THRESH = 30.0
MODEL_TAGS = ["NUX2sNlr5e5_s3000", "base"]


def pick(lst, key, n):
    lst = sorted(lst)
    random.Random(f"1986|{key}").shuffle(lst)
    return lst[:n]


def items():
    """→ {audio: dict(utt, speaker, cell)} — 기쁨A 전부 · 슬픔A · 분노A 화자마다 N_CMP · 무감정 12"""
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    for line in open(TRAIN):
        r = json.loads(line)
        by[r["speaker"]][r["emotion"]].setdefault(r["audio"], r["utt"])
    for line in open(POOL):
        r = json.loads(line)
        by[r["speaker"]]["Neutrality"].setdefault(r["audio"], r["utt"])
    out = {}
    for s, cs in by.items():
        if not cs.get("Happy-A"):
            continue
        for c in CELLS + ["Neutrality"]:
            aud = sorted(cs.get(c, {}))
            if c == "Neutrality":
                aud = pick(aud, f"{s}|Neutrality", N_NEU)            # angryB_pitch.pick 과 같은 키 · 같은 수
            elif c != "Happy-A":
                aud = pick(aud, f"{s}|{c}|happyA_cmp", N_CMP)
            for p in aud:
                out[p] = dict(utt=cs[c][p], speaker=s, cell=c)
    return out


def cmd_score(a):
    sys.path.insert(0, str(H / "q3tts_koft"))
    from ser.models import dim_infer, e2v_infer, load_16k, load_dim, load_e2v
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "ser.json"
    res = json.load(open(f)) if f.exists() else {}
    It = items()
    json.dump(It, open(OUT / "items.json", "w"), ensure_ascii=False)
    todo = [p for p in It if p not in res]
    print("문항", len(It), "· 할 것", len(todo), dict(collections.Counter(v["cell"] for v in It.values())), flush=True)
    e2v, dim = load_e2v(), load_dim()
    t0 = time.time()
    for i, p in enumerate(todo):
        w = load_16k(p)
        labs, sc = e2v_infer(e2v, w)
        d = dict(zip(labs, sc))
        dm = dim_infer(dim, w)
        res[p] = dict(top=labs[int(np.argmax(sc))], **{k: float(d.get(k, 0)) for k in ("happy", "neutral", "sad", "angry", "surprised")},
                      arousal=float(dm.get("arousal", np.nan)), valence=float(dm.get("valence", np.nan)))
        if (i + 1) % 1000 == 0:
            json.dump(res, open(f, "w"))
            print(f"  {i + 1}/{len(todo)} · {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(f, "w"))
    print("✅", len(res))


def cmd_f0(a):
    import angryB_pitch as AB
    It = json.load(open(OUT / "items.json"))
    f = OUT / "f0.json"
    F = json.load(open(f)) if f.exists() else {}
    with Pool(int(a.procs)) as pool:
        neu = [p for p, v in It.items() if v["cell"] == "Neutrality" and p not in F]
        F.update({p: [v, vs] for p, v, vs in pool.map(AB.f0_med, [(p, 60.0, 800.0) for p in neu])})
        nf = {}
        for p, v in It.items():
            if v["cell"] == "Neutrality" and F[p][1] >= MIN_VOICED and np.isfinite(F[p][0]):
                nf.setdefault(v["speaker"], []).append(F[p][0])
        nf = {s: float(np.median(x)) for s, x in nf.items()}
        jobs = [(p, nf[v["speaker"]] / 2, nf[v["speaker"]] * 2 ** (F0_UP / 12)) for p, v in It.items()
                if v["cell"] != "Neutrality" and v["speaker"] in nf and p not in F]
        print("F0", len(jobs), flush=True)
        F.update({p: [v, vs] for p, v, vs in pool.map(AB.f0_med, jobs, chunksize=16)})
    F["_neutral_f0"] = nf
    json.dump(F, open(f, "w"))
    print("✅ 화자 무감정 F0", len(nf))


def neu_like(r):
    return r["top"] == "neutral" and r["happy"] < NEU_LIKE_P


def cmd_report(a):
    import cell_drift as CD
    It = json.load(open(OUT / "items.json"))
    S = json.load(open(OUT / "ser.json"))
    F = json.load(open(OUT / "f0.json"))
    nf = F["_neutral_f0"]
    spk = sorted({v["speaker"] for v in It.values()})
    base = {}
    for s in spk:
        ns = [S[p] for p, v in It.items() if v["speaker"] == s and v["cell"] == "Neutrality" and p in S]
        if ns:
            base[s] = dict(arousal=float(np.mean([r["arousal"] for r in ns])), valence=float(np.mean([r["valence"] for r in ns])), n=len(ns))

    def st(p, s):
        if s not in nf or p not in F or F[p][1] < MIN_VOICED or not np.isfinite(F[p][0]):
            return float("nan")
        return 12 * math.log2(F[p][0] / nf[s])

    def agg(ps, s=None):
        rs = [S[p] for p in ps]
        sp = [It[p]["speaker"] for p in ps]
        tops = collections.Counter(r["top"] for r in rs)
        da = [r["arousal"] - base[x]["arousal"] for r, x in zip(rs, sp) if x in base]
        dv = [r["valence"] - base[x]["valence"] for r, x in zip(rs, sp) if x in base]
        sts = [v for v in (st(p, x) for p, x in zip(ps, sp)) if np.isfinite(v)]
        n = len(rs)
        return dict(n=n, neu_like=100 * np.mean([neu_like(r) for r in rs]), top_neutral=100 * tops["neutral"] / n, top_happy=100 * tops["happy"] / n,
                    top_other=100 * (n - tops["neutral"] - tops["happy"]) / n, top_dist={k: round(100 * v / n, 1) for k, v in tops.most_common()},
                    p_happy=float(np.mean([r["happy"] for r in rs])), p_happy_med=float(np.median([r["happy"] for r in rs])),
                    d_arousal=float(np.mean(da)) if da else None, d_valence=float(np.mean(dv)) if dv else None,
                    arousal=float(np.mean([r["arousal"] for r in rs])), valence=float(np.mean([r["valence"] for r in rs])),
                    st_med=float(np.median(sts)) if sts else None, n_f0=len(sts))
    cells = {c: agg([p for p, v in It.items() if v["cell"] == c and p in S]) for c in CELLS + ["Neutrality"]}
    # 비교 칸과 같은 표본 크기(화자마다 60)로 뽑은 기쁨A 도(표본 차이 확인용)
    ha60 = [p for s in spk for p in pick([p for p, v in It.items() if v["speaker"] == s and v["cell"] == "Happy-A" and p in S], f"{s}|Happy-A|happyA_cmp", N_CMP)]
    cells["Happy-A(화자당 60)"] = agg(ha60)
    by_spk = {}
    for s in spk:
        row = {}
        for c in CELLS:
            ps = [p for p, v in It.items() if v["speaker"] == s and v["cell"] == c and p in S]
            if ps:
                row[c] = agg(ps)
        by_spk[s] = row
    # 화자 가중(화자마다 같은 무게) 기쁨A
    ha_spk = [v["Happy-A"]["neu_like"] for v in by_spk.values() if "Happy-A" in v]
    judge = cells["Happy-A"]["neu_like"]
    # 모델 쪽
    model = {}
    P = json.load(open(D / "plan.json"))
    for t in MODEL_TAGS:
        ser = json.load(open(D / f"ser_{t}.json"))
        m = {}
        for c in CELLS + ["Neutrality"]:
            xs = [x for x in P["items"] if x["emo"] == c and x["id"] in ser]
            rs = [ser[x["id"]] for x in xs]
            m[c] = dict(n=len(rs), top_happy=100 * np.mean([r["top"] == "happy" for r in rs]), top_neutral=100 * np.mean([r["top"] == "neutral" for r in rs]),
                        neu_like=100 * np.mean([neu_like(r) for r in rs]), p_happy=float(np.mean([r["happy"] for r in rs])),
                        arousal=float(np.mean([r["arousal"] for r in rs])), valence=float(np.mean([r["valence"] for r in rs])))
        # 모델 출력 F0(REF 대비 반음 · iv3_compare 와 같은 규칙: REF 60~800Hz · 출력은 [REF/2, REF +20반음])
        mf = OUT / "model_f0.json"
        MF = json.load(open(mf)) if mf.exists() else {}
        jobs = []
        import angryB_pitch as AB
        if "REF" not in MF:
            with Pool(16) as pool:
                MF["REF"] = {s: v for s, (_, v, _) in zip(P["spk"], pool.map(AB.f0_med, [(v["ref"], 60.0, 800.0) for v in P["spk"].values()]))}
        for x in P["items"]:
            k = f"{t}|{x['id']}"
            if x["emo"] in CELLS + ["Neutrality"] and k not in MF:
                fr = MF["REF"][x["tgt"]]
                jobs.append((k, (str(D / "wav" / t / f"{x['id']}.wav"), fr / 2, fr * 2 ** (F0_UP / 12))))
        if jobs:
            with Pool(32) as pool:
                for (k, _), (_, v, vs) in zip(jobs, pool.map(AB.f0_med, [j for _, j in jobs])):
                    MF[k] = [v, vs]
            json.dump(MF, open(mf, "w"))
        for c in CELLS + ["Neutrality"]:
            sts = [12 * math.log2(MF[f"{t}|{x['id']}"][0] / MF["REF"][x["tgt"]]) for x in P["items"] if x["emo"] == c
                   and MF[f"{t}|{x['id']}"][1] >= MIN_VOICED and np.isfinite(MF[f"{t}|{x['id']}"][0])]
            m[c]["st_med_vs_ref"] = float(np.median(sts)) if sts else None
        try:
            dr = CD.cells(t)
            m["drift"] = {c: dr[c] for c in dr} if isinstance(dr, dict) else None
        except Exception as e:  # noqa: BLE001
            m["drift"] = f"없음({e})"
        model[t] = m
    summ = dict(judge=dict(rule=f"e2v top = neutral 그리고 happy < {NEU_LIKE_P}", neu_like_pct=judge, threshold=THRESH, run_s5=bool(judge >= THRESH),
                           speaker_weighted=float(np.mean(ha_spk)), speakers_over30=int(sum(x >= THRESH for x in ha_spk)), n_speakers=len(ha_spk)),
                cells=cells, model=model, neutral_baseline_speakers=len(base), n_speakers=len(spk))
    json.dump(dict(summary=summ, by_speaker=by_spk), open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1, default=float)
    # 화자별 표(md)
    L = ["| 화자 | 기쁨A 녹음 | 무감정처럼 | e2v happy 1위 | happy 확률 | arousal 차 | valence 차 | F0 반음 | 슬픔A 무감정처럼 | 분노A 무감정처럼 |", "|---|---|---|---|---|---|---|---|---|---|"]
    f1 = lambda v, f="{:.0f}%": "–" if v is None else f.format(v)  # noqa: E731
    for s in sorted(by_spk, key=lambda s: -by_spk[s].get("Happy-A", {}).get("neu_like", -1)):
        h = by_spk[s].get("Happy-A")
        if not h:
            continue
        L.append(f"| {s} | {h['n']} | {h['neu_like']:.0f}% | {h['top_happy']:.0f}% | {h['p_happy']:.2f} | {f1(h['d_arousal'], '{:+.3f}')} | {f1(h['d_valence'], '{:+.3f}')} | "
                 f"{f1(h['st_med'], '{:+.1f}')} | {f1(by_spk[s].get('Sad-A', {}).get('neu_like'))} | {f1(by_spk[s].get('Angry-A', {}).get('neu_like'))} |")
    (OUT / "by_speaker.md").write_text("\n".join(L) + "\n")
    for c, v in cells.items():
        print(f"{c:20s} n {v['n']:5d} · 무감정처럼 {v['neu_like']:.1f}% · top neutral {v['top_neutral']:.0f}% happy {v['top_happy']:.0f}% 그밖 {v['top_other']:.0f}% · "
              f"p_happy {v['p_happy']:.2f} · Δaro {v['d_arousal'] if v['d_arousal'] is None else round(v['d_arousal'], 3)} · Δval {v['d_valence'] if v['d_valence'] is None else round(v['d_valence'], 3)} · F0 {v['st_med']}")
    print("판정", summ["judge"])
    for t, m in model.items():
        print(t, {c: (round(v["top_happy"]), round(v["neu_like"]), round(v["p_happy"], 2), v["st_med_vs_ref"] and round(v["st_med_vs_ref"], 1)) for c, v in m.items() if c != "drift"}, "drift", m["drift"] if not isinstance(m["drift"], dict) else {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m["drift"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score", "f0", "report"])
    ap.add_argument("--procs", default=64)
    a = ap.parse_args()
    {"score": cmd_score, "f0": cmd_f0, "report": cmd_report}[a.cmd](a)
