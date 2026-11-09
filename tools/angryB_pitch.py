#!/usr/bin/env python
"""분노B 정답 녹음 높이 조사 — 학습 데이터의 분노B 는 원래 무감정보다 얼마나 높은가 (2026-11-09 · 지시문 q3tts 세션 11/08 23:25 §4)

물음: 11/08 시험 A 에서 문구를 바꿔도 모델 분노B 가 REF 대비 +3.5반음 남았다(지금 문구 C0 은 +6.7).
      학습 데이터의 분노B 정답 녹음이 원래 그만큼 높아서인가?
대상: A7NUX2sN 학습셋(utt 중복 제거)에서 분노B 정답이 있는 화자 전부. 비교 칸 = 분노A · 기쁨B · 무감정.
잴 것: 화자마다 그 칸 녹음 F0 중앙값 − 같은 화자 무감정 녹음 F0 중앙값(반음)
  - 무감정 녹음 = 학습셋 Neutrality 행 + A7N/ref_pool(무감정 ref 풀). 화자마다 칸마다 최대 N_PER 개(seed 1986 · 문자열 키)
  - 범위 규칙은 시험 A(tools/instr_test.py)와 같음: 무감정은 60~800Hz · 감정 칸은 화자마다 [무감정 F0 / 2, 무감정 F0 + 20반음]
  - 유성 구간이 0.3초보다 짧은 녹음은 뺌
모델 쪽(같은 규칙): D 재측정 음원(A7NUX2sN s3000 · 16명 · 칸마다 2문장)의 REF 대비 높이도 같이 잼.
    ~/miniforge3/envs/voxcpm/bin/python tools/angryB_pitch.py run      # → tmp_listen/vox_koft/angryB_pitch/{f0.json,summary.json}
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

H = Path.home() / "workspaces"
PRE = H / "preprocessed/vox_koft"
TRAIN = PRE / "A7NUX2sN/train.jsonl"
POOL = PRE / "A7N/ref_pool.jsonl"
D = H / "tmp_listen/vox_koft/d_eval"
OUT = H / "tmp_listen/vox_koft/angryB_pitch"
CELLS = ["Angry-B", "Angry-A", "Happy-B"]
N_PER = 12
F0_UP = 20
MIN_VOICED = 0.3
MODEL_TAG = "NUX2sNlr5e5_s3000"


def f0_med(job):
    import librosa
    path, fmin, fmax = job
    y, sr = librosa.load(str(path), sr=16000)
    f0, vf, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=160)
    v = f0[vf & np.isfinite(f0)]
    return str(path), (float(np.median(v)) if len(v) else float("nan")), float(len(v) * 0.01)


def st(a, b):
    return 12 * math.log2(a / b)


def src_of(s: str) -> str:
    return s.split("_")[0]


def pick(lst, key):
    lst = sorted(lst)
    random.Random(f"1986|{key}").shuffle(lst)
    return lst[:N_PER]


def cmd_run(a):
    rows, seen = defaultdict(lambda: defaultdict(list)), set()
    for line in open(TRAIN):
        r = json.loads(line)
        if r["utt"] in seen:
            continue
        seen.add(r["utt"])
        rows[r["speaker"]][r["emotion"]].append(r["audio"])
    for line in open(POOL):
        r = json.loads(line)
        if r["audio"] not in rows[r["speaker"]]["Neutrality"]:
            rows[r["speaker"]]["Neutrality"].append(r["audio"])
    spks = sorted(s for s, v in rows.items() if v["Angry-B"])
    no_neu = [s for s in spks if not rows[s]["Neutrality"]]
    use = [s for s in spks if rows[s]["Neutrality"]]
    print(f"분노B 화자 {len(spks)} · 무감정 녹음 있음 {len(use)} · 없음 {len(no_neu)} {no_neu}")
    OUT.mkdir(parents=True, exist_ok=True)
    P = json.load(open(D / "plan.json"))
    with Pool(int(a.procs)) as pool:
        # 1) 무감정(넓은 범위)
        neu_jobs = {s: pick(rows[s]["Neutrality"], f"{s}|Neutrality") for s in use}
        ref_jobs = {s: v["ref"] for s, v in P["spk"].items()}
        res = dict((p, (f, v)) for p, f, v in pool.map(
            f0_med, [(p, 60.0, 800.0) for ps in neu_jobs.values() for p in ps] + [(p, 60.0, 800.0) for p in ref_jobs.values()]))
        neu = {}
        for s, ps in neu_jobs.items():
            fs = [res[p][0] for p in ps if res[p][1] >= MIN_VOICED and np.isfinite(res[p][0])]
            if fs:
                neu[s] = float(np.median(fs))
        # 2) 감정 칸(화자 범위)
        jobs = []
        for s, f in neu.items():
            for c in CELLS:
                jobs += [(p, f / 2, f * 2 ** (F0_UP / 12)) for p in pick(rows[s][c], f"{s}|{c}")]
        model = {}
        for x in P["items"]:
            if x["emo"] in CELLS + ["Neutrality"]:
                fr = res[str(ref_jobs[x["tgt"]])][0]
                jobs.append((str(D / "wav" / MODEL_TAG / f"{x['id']}.wav"), fr / 2, fr * 2 ** (F0_UP / 12)))
                model[x["id"]] = (x["tgt"], x["emo"], jobs[-1][0])
        res.update(dict((p, (f, v)) for p, f, v in pool.map(f0_med, jobs)))
    # 데이터: 화자 × 칸
    data, files = {}, defaultdict(list)
    for s, f in neu.items():
        data[s] = {"neutral_f0": f, "src": src_of(s)}
        for c in CELLS:
            ps = [p for p in pick(rows[s][c], f"{s}|{c}") if res[p][1] >= MIN_VOICED and np.isfinite(res[p][0])]
            if not ps:
                continue
            sts = [st(res[p][0], f) for p in ps]
            data[s][c] = {"st": float(np.median(sts)), "n": len(ps), "n_all": len(rows[s][c])}
            files[c] += sts
    # 모델: REF 대비
    mod = defaultdict(list)
    for i, (s, e, p) in model.items():
        fr = res[str(ref_jobs[s])][0]
        if res[p][1] >= MIN_VOICED and np.isfinite(res[p][0]):
            mod[e].append({"id": i, "spk": s, "st": st(res[p][0], fr), "src": P["spk"][s]["src"]})
    summ = {"data": {}, "model": {}, "n_spk": len(neu), "no_neutral": no_neu, "n_per": N_PER}
    for c in CELLS:
        v = sorted((d[c]["st"], s) for s, d in data.items() if c in d)
        arr = np.array([x for x, _ in v])
        fa = np.array(files[c])
        summ["data"][c] = {"n_spk": len(v), "min": float(arr.min()), "q25": float(np.percentile(arr, 25)), "median": float(np.median(arr)),
                           "q75": float(np.percentile(arr, 75)), "max": float(arr.max()), "mean": float(arr.mean()),
                           "n_files": int(len(fa)), "file_median": float(np.median(fa)), "over8_files": float(100 * np.mean(fa > 8)),
                           "over8_spk": float(100 * np.mean(arr > 8)),
                           "top5": [(s, round(x, 2)) for x, s in v[::-1][:5]], "bottom5": [(s, round(x, 2)) for x, s in v[:5]],
                           "by_src": {g: float(np.median([x for x, s in v if src_of(s) == g])) for g in sorted({src_of(s) for _, s in v})}}
    for e, lst in mod.items():
        arr = np.array([x["st"] for x in lst])
        summ["model"][e] = {"n": len(lst), "median": float(np.median(arr)), "mean": float(arr.mean()),
                            "over8": float(100 * np.mean(arr > 8)),
                            "by_src": {g: float(np.median([x["st"] for x in lst if x["src"] == g])) for g in ("542", "unseen8") if any(x["src"] == g for x in lst)}}
    json.dump({"data": data, "model": mod, "neutral_f0": neu}, open(OUT / "f0.json", "w"), ensure_ascii=False, indent=1)
    json.dump(summ, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1)
    for c, v in summ["data"].items():
        print(f"[데이터] {c:8s} 화자 {v['n_spk']} · 최저 {v['min']:+.1f} · 중앙 {v['median']:+.1f} · 최고 {v['max']:+.1f} · 평균 {v['mean']:+.1f}"
              f" · +8 넘는 녹음 {v['over8_files']:.0f}% · 화자 {v['over8_spk']:.0f}% · 출처별 {v['by_src']}")
    for e, v in summ["model"].items():
        print(f"[모델 {MODEL_TAG}] {e:10s} n {v['n']} · 중앙 {v['median']:+.1f} · 평균 {v['mean']:+.1f} · +8 넘음 {v['over8']:.0f}% · {v['by_src']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--procs", default=48)
    a = ap.parse_args()
    {"run": cmd_run}[a.cmd](a)
