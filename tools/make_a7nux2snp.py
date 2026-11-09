#!/usr/bin/env python
"""A7NUX2sNp — B칸 고르기 판: A7NUX2sN 에서 분노B · 기쁨B 정답 중 "너무 높은 음" · "갈라진(클리핑) 녹음" 행만 뺌
(2026-11-09 · 지시문 q3tts 세션 11/09 09:30 §1 · 사용자 결정 11/09 09시)

바탕: A7NUX2sN(두 프로젝트 같은 행). 바꾸는 것은 **정답 행 빼기 하나**뿐.
  - ref 배정 · 다른 칸 · 무감정 행 · val_xref 그대로. 같은 녹음이 다른 행의 ref 로 쓰인 경우도 그대로 둠(한 가지만 바꿈).
  - 빠진 행은 채우지 않음(행 수가 줄어든 그대로 학습).
빼는 규칙(칸 = Angry-B · Happy-B, utt 단위 — 그 utt 의 행(ref 2개)을 모두 뺌)
  1. 높은 음: 녹음 F0 중앙값 − 같은 화자 무감정 F0 중앙값 > +8반음
     - 재는 법 = tools/angryB_pitch.py(pyin · 16kHz · 무감정 60~800Hz · 감정은 [무감정/2, 무감정 +20반음] · 유성 0.3초 미만은 판정 불가 → 높이로는 안 뺌)
     - 화자 무감정 F0 = angryB_pitch 와 같은 무감정 12개(seed 1986 · 문자열 키) 중앙값
     - 무감정 녹음이 없는 화자는 높이 규칙을 적용하지 않음(클리핑만)
  2. 클리핑: 원본 정수 샘플에서 "자기 최대값의 98% 이상 · 같은 값 연속" 길이(flat_eq) · 평평한 꼭대기 수(plateau) — clip_stat() 설명
     · 기준은 분포를 보고 정함(build 인자 · 보고서에 적음)
  최소치: 화자 · 칸마다 keep_min = min(n, max(ceil(0.3 n), 20)). 빼는 후보를 다 빼면 그보다 적어지면
          후보 중 (클리핑 아님 먼저, 그다음 F0 낮은 순)으로 되살려 keep_min 을 채움.

  ~/miniforge3/envs/voxcpm/bin/python tools/make_a7nux2snp.py measure        # → tmp_listen/vox_koft/bcell_pick/measure.json (캐시)
  ~/miniforge3/envs/voxcpm/bin/python tools/make_a7nux2snp.py dist           # 클리핑 분포 보기
  ~/miniforge3/envs/voxcpm/bin/python tools/make_a7nux2snp.py build --clip_flat 4
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import angryB_pitch as AB  # noqa: E402

H = Path.home() / "workspaces"
V = H / "preprocessed/vox_koft"
Q = H / "preprocessed/q3tts_koft"
BASE, NAME = "A7NUX2sN", "A7NUX2sNp"
OUT = H / "tmp_listen/vox_koft/bcell_pick"
CELLS = ["Angry-B", "Happy-B"]
ST_MAX = 8.0
CLIP_REL = 0.98
KEEP_FRAC, KEEP_MIN = 0.3, 20


def orig_paths():
    """원본(48kHz 등) 경로: AQ_shared/audio_log.csv 의 src · 없으면(A7NU 71349) 행의 src_audio"""
    import csv
    return {r["utt"]: r["src"] for r in csv.DictReader(open(Q / "AQ_shared/audio_log.csv"))}


def runs(m):
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def clip_stat(path):
    """갈라짐(클리핑) 흔적. 이 데이터는 원본도 최대 진폭이 0.92 FS 정도라(정규화됨) |x| ≥ 0.99 FS 는 0개
    → 정규화 전에 잘렸다면 남는 "자기 최대값 근처의 평평한 꼭대기"를 잼(원본 정수 샘플)
      flat_eq = |x| ≥ 0.98 × 파일 최대 인 곳에서 **같은 값이 연속**된 가장 긴 길이(샘플)
      plateau = |x| ≥ 0.98 × 파일 최대 가 4샘플 이상 이어진 구간 수"""
    import soundfile as sf
    x, sr = sf.read(str(path), always_2d=True, dtype="int16")
    x = x[:, 0].astype(np.int32)
    a = np.abs(x)
    pk = int(a.max())
    hi = a >= CLIP_REL * pk
    L = runs(hi)
    same = hi[1:] & hi[:-1] & (x[1:] == x[:-1])
    Ls = runs(same)
    return dict(flat_eq=int(Ls.max() + 1) if len(Ls) else 0, plateau=int((L >= 4).sum()), peak_fs=pk / 32767, full_scale=int((a >= 32767 * 0.99).sum()), sr0=int(sr))


def job(j):
    utt, p16, psrc, fmin, fmax = j
    f, vs = (float("nan"), 0.0) if fmin is None else AB.f0_med((p16, fmin, fmax))[1:]
    return utt, dict(f0=f, voiced=vs, src0=psrc, **clip_stat(psrc))


def load_rows(root):
    return [json.loads(l) for l in open(root / BASE / "train.jsonl")]


def b_targets(rows):
    t = {}
    for r in rows:
        if r["emotion"] in CELLS and r["utt"] not in t:
            t[r["utt"]] = r
    return t


def neutral_f0(rows, spks, pool):
    """angryB_pitch 와 같은 무감정 12개 · 넓은 범위 · 중앙값"""
    neu = collections.defaultdict(list)
    seen = set()
    for r in rows:
        if r["utt"] in seen:
            continue
        seen.add(r["utt"])
        if r["emotion"] == "Neutrality":
            neu[r["speaker"]].append(r["audio"])
    for line in open(AB.POOL):
        r = json.loads(line)
        if r["audio"] not in neu[r["speaker"]]:
            neu[r["speaker"]].append(r["audio"])
    picks = {s: AB.pick(neu[s], f"{s}|Neutrality") for s in spks if neu[s]}
    res = dict((p, (f, v)) for p, f, v in pool.map(AB.f0_med, [(p, 60.0, 800.0) for ps in picks.values() for p in ps]))
    out = {}
    for s, ps in picks.items():
        fs = [res[p][0] for p in ps if res[p][1] >= AB.MIN_VOICED and np.isfinite(res[p][0])]
        if fs:
            out[s] = float(np.median(fs))
    return out


def cmd_measure(a):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "measure.json"
    M = json.load(open(f)) if f.exists() else {"neutral_f0": {}, "utt": {}}
    rows = load_rows(V)
    T = b_targets(rows)
    spks = sorted({r["speaker"] for r in T.values()})
    with Pool(int(a.procs)) as pool:
        if not M["neutral_f0"]:
            M["neutral_f0"] = neutral_f0(rows, spks, pool)
            json.dump(M, open(f, "w"), ensure_ascii=False)
        nf = M["neutral_f0"]
        old = json.load(open(AB.OUT / "f0.json"))["neutral_f0"]
        same = [s for s in old if s in nf and abs(old[s] - nf[s]) > 1e-6]
        print(f"화자 {len(spks)} · 무감정 F0 있음 {len(nf)} · angryB_pitch 와 다른 화자 {same}", flush=True)
        jobs, OP = [], orig_paths()
        for u, r in T.items():
            if u in M["utt"]:
                continue
            fr = nf.get(r["speaker"])
            jobs.append((u, r["audio"], OP.get(u, r["src_audio"]), (fr / 2) if fr else None, (fr * 2 ** (AB.F0_UP / 12)) if fr else None))
        print("잴 녹음", len(jobs), "(캐시", len(M["utt"]), ")", flush=True)
        t0 = time.time()
        for i, (u, d) in enumerate(pool.imap_unordered(job, jobs, chunksize=16)):
            M["utt"][u] = d
            if (i + 1) % 2000 == 0:
                json.dump(M, open(f, "w"), ensure_ascii=False)
                print(f"  {i + 1}/{len(jobs)} · {time.time() - t0:.0f}s", flush=True)
    json.dump(M, open(f, "w"), ensure_ascii=False)
    print("✅", len(M["utt"]))


def cmd_dist(a):
    M = json.load(open(OUT / "measure.json"))
    T = b_targets(load_rows(V))
    for c in CELLS + ["all"]:
        ds = [M["utt"][u] for u, r in T.items() if c in ("all", r["emotion"])]
        fe = np.array([d["flat_eq"] for d in ds]); pl = np.array([d["plateau"] for d in ds]); pk = np.array([d["peak_fs"] for d in ds])
        print(f"[{c}] n {len(ds)} · 원본 최대 진폭 분위(50/90/99/최대) {np.percentile(pk, [50, 90, 99, 100]).round(3)} · 0.99 FS 넘는 샘플 있는 녹음 {sum(d['full_scale'] > 0 for d in ds)}")
        print("    flat_eq 분위(50/90/99/99.9/최대)", np.percentile(fe, [50, 90, 99, 99.9, 100]), " plateau", np.percentile(pl, [50, 90, 99, 99.9, 100]))
        print("    " + " · ".join(f"flat_eq≥{t} {100 * np.mean(fe >= t):.2f}%({(fe >= t).sum()})" for t in (2, 3, 4, 6, 8)))
        print("    " + " · ".join(f"plateau≥{t} {100 * np.mean(pl >= t):.2f}%({(pl >= t).sum()})" for t in (5, 10, 20, 40)))
    print("원본 sr", dict(collections.Counter(d["sr0"] for d in M["utt"].values())))


def decide(M, T, clip_flat, clip_plateau):
    nf = M["neutral_f0"]
    by = collections.defaultdict(list)
    for u, r in T.items():
        by[(r["speaker"], r["emotion"])].append(u)
    drop, info = set(), {}
    for (s, c), us in sorted(by.items()):
        n = len(us)
        keep_min = min(n, max(math.ceil(KEEP_FRAC * n), KEEP_MIN))
        st, high, clip, cand = {}, set(), set(), []
        for u in us:
            d = M["utt"][u]
            ok = s in nf and d["voiced"] >= AB.MIN_VOICED and np.isfinite(d["f0"])
            st[u] = AB.st(d["f0"], nf[s]) if ok else float("nan")
            if ok and st[u] > ST_MAX:
                high.add(u)
            if d["flat_eq"] >= clip_flat or d["plateau"] >= clip_plateau:
                clip.add(u)
        cand = high | clip
        n_keep = n - len(cand)
        back = []
        if n_keep < keep_min:
            order = sorted(cand, key=lambda u: (u in clip, st[u] if np.isfinite(st[u]) else -1e9, u))
            back = order[:keep_min - n_keep]
        dr = cand - set(back)
        drop |= dr
        vals = [v for v in st.values() if np.isfinite(v)]
        kept_vals = [st[u] for u in us if u not in dr and np.isfinite(st[u])]
        info[f"{s}|{c}"] = dict(speaker=s, cell=c, n=n, keep_min=keep_min, high=len(high), clip=len(clip), both=len(high & clip),
                                restored=len(back), dropped=len(dr), kept=n - len(dr), has_neutral=s in nf,
                                n_f0=len(vals), st_med_before=float(np.median(vals)) if vals else None,
                                st_med_after=float(np.median(kept_vals)) if kept_vals else None,
                                over8_before=float(100 * np.mean(np.array(vals) > 8)) if vals else None,
                                over8_after=float(100 * np.mean(np.array(kept_vals) > 8)) if kept_vals else None)
    return drop, info


def cell_dist(M, T, keep, nf):
    out = {}
    for c in CELLS:
        a = [AB.st(M["utt"][u]["f0"], nf[r["speaker"]]) for u, r in T.items() if r["emotion"] == c and u in keep and r["speaker"] in nf
             and M["utt"][u]["voiced"] >= AB.MIN_VOICED and np.isfinite(M["utt"][u]["f0"])]
        a = np.array(a)
        out[c] = dict(n_f0=len(a), q25=float(np.percentile(a, 25)), median=float(np.median(a)), q75=float(np.percentile(a, 75)),
                      q90=float(np.percentile(a, 90)), over8=float(100 * np.mean(a > 8)), mean=float(a.mean()))
    return out


def cmd_build(a):
    M = json.load(open(OUT / "measure.json"))
    rows = {root: load_rows(root) for root in (V, Q)}
    T = b_targets(rows[V])
    assert len(M["utt"]) >= len(T) and all(u in M["utt"] for u in T)
    drop, info = decide(M, T, a.clip_flat, a.clip_plateau)
    nf = M["neutral_f0"]
    before = cell_dist(M, T, set(T), nf)
    after = cell_dist(M, T, set(T) - drop, nf)
    new = {root: [r for r in rs if r["utt"] not in drop] for root, rs in rows.items()}
    # 검사: 두 프로젝트 같은 행 · 뺀 것은 B칸 정답만 · ref 그대로
    chk = dict(
        rows_differ=int(len(new[V]) != len(new[Q]) or any(x["utt_x"] != y["utt_x"] for x, y in zip(new[V], new[Q]))),
        dropped_non_b=sum(1 for r in rows[V] if r["utt"] in drop and r["emotion"] not in CELLS),
        b_cell_vanished=sum(1 for v in info.values() if v["kept"] == 0),
        under_keep_min=sum(1 for v in info.values() if v["kept"] < v["keep_min"]),
    )
    emo = collections.Counter(r["emotion"] for r in new[V])
    emo0 = collections.Counter(r["emotion"] for r in rows[V])
    ref_as = collections.Counter(r["ref_utt"] for r in new[V])
    agg = {c: dict(targets=sum(v["n"] for v in info.values() if v["cell"] == c),
                   high=sum(v["high"] for v in info.values() if v["cell"] == c),
                   clip=sum(v["clip"] for v in info.values() if v["cell"] == c),
                   both=sum(v["both"] for v in info.values() if v["cell"] == c),
                   restored=sum(v["restored"] for v in info.values() if v["cell"] == c),
                   dropped=sum(v["dropped"] for v in info.values() if v["cell"] == c),
                   speakers=sum(1 for v in info.values() if v["cell"] == c),
                   speakers_no_neutral=sorted(v["speaker"] for v in info.values() if v["cell"] == c and not v["has_neutral"]),
                   speakers_restored=sum(1 for v in info.values() if v["cell"] == c and v["restored"] > 0)) for c in CELLS}
    info_all = dict(
        name=NAME, created_kst=time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600)),
        why="B칸 고르기 판(지시문 q3tts 세션 11/09 09:30 §1 · 사용자 결정 11/09 09시): 분노B · 기쁨B 의 너무 높은 음(+8반음 초과) · 갈라진 녹음 정답 행만 뺌",
        rule=dict(base=BASE, cells=CELLS, st_max=ST_MAX, f0="tools/angryB_pitch.py 와 같음(pyin 16kHz · 화자 범위 · 유성 0.3초 미만 판정 불가)",
                  clip=dict(rel_level=CLIP_REL, flat_eq_ge=a.clip_flat, plateau_ge=a.clip_plateau, on="원본(audio_log.csv src · 없으면 src_audio) 정수 샘플"),
                  keep_min=f"min(n, max(ceil({KEEP_FRAC} n), {KEEP_MIN})) · 되살리는 순서 = 클리핑 아님 먼저 → F0 낮은 순",
                  refs="그대로(같은 녹음이 다른 행의 ref 인 경우도 그대로)", refill="없음", val_xref=f"{BASE}/val_xref.jsonl 그대로", seed=1986),
        counts=dict(rows=len(new[V]), rows_base=len(rows[V]), rows_dropped=len(rows[V]) - len(new[V]),
                    targets_dropped=len(drop), by_cell=agg, target_emotion=dict(emo), target_emotion_base=dict(emo0),
                    share_neutral=round(emo["Neutrality"] / len(new[V]), 4),
                    dropped_utts_still_used_as_ref=sum(1 for u in drop if u in ref_as),
                    rows_with_dropped_utt_as_ref=sum(ref_as[u] for u in drop)),
        pitch_before=before, pitch_after=after, checks=chk, code=str(Path(__file__).resolve()))
    assert all(v == 0 for v in chk.values()), chk
    if a.dry:
        print(json.dumps(dict(counts=info_all["counts"], before=before, after=after), ensure_ascii=False, indent=1))
        return
    for root in (V, Q):
        d = root / NAME
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "train.jsonl", "w", encoding="utf-8") as f:
            for r in new[root]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        shutil.copy(root / BASE / "val_xref.jsonl", d / "val_xref.jsonl")
        json.dump(dict(info_all, project=("vox_koft · 16kHz" if root == V else "q3tts_koft · 24kHz 원본 경로")),
                  open(d / "dataset_info.json", "w"), ensure_ascii=False, indent=1)
        (d / "_READY").write_text(info_all["created_kst"] + "\n")
    json.dump(dict(speaker_cell=info, dropped=sorted(drop)), open(OUT / "decide.json", "w"), ensure_ascii=False, indent=1)
    # 귀로 확인할 클리핑 · 높은 음 예시
    ex = dict(clip=[], high=[])
    for u in sorted(drop):
        d, r = M["utt"][u], T[u]
        if d["flat_eq"] >= a.clip_flat or d["plateau"] >= a.clip_plateau:
            ex["clip"].append((d["flat_eq"], d["plateau"], u, r["emotion"], r["speaker"], d["src0"], r["audio"]))
        elif r["speaker"] in nf:
            ex["high"].append((AB.st(d["f0"], nf[r["speaker"]]), u, r["emotion"], r["speaker"], d["src0"], r["audio"]))
    ex = {k: sorted(v, reverse=True)[:12] for k, v in ex.items()}
    json.dump(ex, open(OUT / "examples.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(counts=info_all["counts"], before=before, after=after, checks=chk), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["measure", "dist", "build"])
    ap.add_argument("--procs", default=96)
    ap.add_argument("--clip_flat", type=int, default=4, help="flat_eq 이상이면 갈라짐")
    ap.add_argument("--clip_plateau", type=int, default=10**9, help="plateau 이상이면 갈라짐(기본 끔)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    {"measure": cmd_measure, "dist": cmd_dist, "build": cmd_build}[a.cmd](a)
