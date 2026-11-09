#!/usr/bin/env python
"""REF 유사도 조사 — "모델 출력이 그 사람의 진짜 녹음과 비슷한가"를 칸마다 공정하게 (2026-11-09 · 지시문 q3tts 세션 11/09 09:30 §4)

왜: 지금의 "목표 유사도"는 무감정 녹음과만 비교. 진짜 사람도 화내면 목소리가 달라지므로, 같은 칸 진짜 녹음끼리 비교해야 공정
    (q3tts 의 "공정한 잣대"와 같은 생각). **새 합성 없음** — D 재측정 음원 임베딩(d_eval/vspk/emb_<tag>.npz)을 그대로 씀.
잣대: WavLM-large(w) · ERes2Net(e) — d_eval sv 와 같은 q3ko.eval.sv.load_embedder · 코사인
진짜 녹음(pick)
  - unseen8 8명(71349 원본 metadata.csv): 칸마다 최대 10개 · 강도 1 → A · 2/3 → B · 무감정 = 강도 0 · 투표 평균 ≥ 4.0 · 2~10초
    · REF · d_eval 문장 · 감정 ref 판 ref 와 겹치는 것 제외 · seed 1986(random.Random(f"1986|refsim|화자|칸"))
  - 542 8명: 무감정만 · 542/eval.jsonl 의 그 화자 녹음 중 REF(긴 ref 에 이어 붙인 문장) 와 글이 겹치는 것 제외 · 최대 10개
  - 다른 화자 후보(1위 비율): 71349 화자 중 **어느 학습셋에도 없는** 화자(unseen8 제외) · 같은 규칙으로 칸마다 최대 10개
잴 것
  (가) 같은 칸(unseen8): 모델 칸 c 출력 ↔ 그 사람 진짜 칸 c 녹음 N개(1 · 3 · 5 · 10) 코사인 평균 · 사람 천장 = 진짜 칸 c 녹음 하나 ↔ 나머지 평균
       · 1위 비율 = 같은 성별 후보들의 같은 칸 진짜 녹음 평균(중심) 중 목표가 1위인가
  (나) REF 개수(16명): 모델 출력 ↔ 그 사람 무감정 녹음 N개 평균 → 모델 순위가 N 에 따라 바뀌는가
  (다) 칸 쌍 표: 모델마다 같은 화자 7 × 7 칸 평균 코사인 · 진짜 녹음(unseen8) 같은 표 · 사람보다 더 멀어지는 칸 쌍
  (라) 기쁨A 후보 섞기: NUX2sN s3000 의 6칸(기쁨A 제외) ↔ 기쁨A 후보 출력 · 낙차 = 후보 − NUX2sN 자신

  ~/miniforge3/envs/voxcpm/bin/python tools/ref_sim.py pick
  CUDA_VISIBLE_DEVICES=3 ~/miniforge3/envs/q3tts/bin/python tools/ref_sim.py emb
  ~/miniforge3/envs/voxcpm/bin/python tools/ref_sim.py analyze [--extra_tags NUX2sNplr5e5_s3000]
  CUDA_VISIBLE_DEVICES=3 ~/miniforge3/envs/voxcpm/bin/python tools/ref_sim.py repro   # 페이지 음원 = 노트북 엔진 소리인지
  ~/miniforge3/envs/voxcpm/bin/python tools/ref_sim.py page                            # → tmp_listen/vox_koft/ref_check/index.html
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

H = Path.home() / "workspaces"
D = H / "tmp_listen/vox_koft/d_eval"
VS = D / "vspk"
OUT = H / "tmp_listen/vox_koft/ref_sim"
META = H / "ourdata/aihub_71349/meta/metadata.csv"
RAW = Path("/data1/rohoon/aihub_71349")
E542 = H / "preprocessed/q3tts_koft/542/eval.jsonl"
EMO7 = ["Neutrality", "Happy-A", "Happy-B", "Sad-A", "Sad-B", "Angry-A", "Angry-B"]
KO = {"Neutrality": "중립", "Happy-A": "기쁨A", "Happy-B": "기쁨B", "Sad-A": "슬픔A", "Sad-B": "슬픔B", "Angry-A": "분노A", "Angry-B": "분노B"}
EN = {"Neutrality": "Neu", "Happy-A": "HapA", "Happy-B": "HapB", "Sad-A": "SadA", "Sad-B": "SadB", "Angry-A": "AngA", "Angry-B": "AngB"}
KO2EN = {"기쁨": "Happy", "슬픔": "Sad", "분노": "Angry"}
NMAX, VOTES = 10, 4.0
NS = [1, 3, 5, 10]
TAGS = ["NUX2sNlr5e5_s3000", "NUX2sNlr5e5_s4500", "A7NX3_s4500", "A7NX3_s2250", "X2Rs6lr5e5_s4500", "X2Rs6lr5e5_s2500",
        "X2Rs01lr5e5_s4000", "NUX2slr5e5_s4000", "base"]
HAPPY_CAND = ["NUX2sNlr5e5_s3000", "A7NX3_s4500", "A7NX3_s2250", "X2Rs6lr5e5_s2500"]
MAIN = "NUX2sNlr5e5_s3000"
clean = lambda t: re.sub(r"[^가-힣a-zA-Z0-9]", "", t or "")  # noqa: E731


def jl(p):
    return json.load(open(p))


def cell_of_meta(m):
    if m["emotion"] == "무감정" and m["intensity"] == "0":
        return "Neutrality"
    if m["emotion"] in KO2EN and m["intensity"] in ("1", "2", "3"):
        return f"{KO2EN[m['emotion']]}-{'A' if m['intensity'] == '1' else 'B'}"
    return None


def used_aihub():
    used = set()
    for f in (H / "preprocessed/vox_koft").glob("*/train.jsonl"):
        for line in open(f):
            s = json.loads(line)["speaker"]
            if s.startswith("aihub_"):
                used.add(int(s.split("_")[1]))
    return used


def cmd_pick(a):
    P = jl(D / "plan.json")
    bad_txt = {clean(x["text"]) for x in P["sents"]} | {clean(x["text"]) for x in P["items"]}
    ef = H / "tmp_listen/vox_koft/d_eval_eref/plan.json"
    eref = {x["ref_utt"] for x in jl(ef)["items"]} if ef.exists() else set()
    un = {s: v for s, v in P["spk"].items() if v["src"] == "unseen8"}
    used = used_aihub()
    meta = list(csv.DictReader(open(META, encoding="utf-8")))
    rid_g = {int(m["reciter_id"]): m["reciter_gender"].lower() for m in meta}
    tgt_ids = {int(s.split("_")[1]) for s in un}
    distract = sorted(r for r in rid_g if r not in used and r not in tgt_ids)
    ref_utt = {s: Path(v["ref"]).stem for s, v in P["spk"].items()}
    by = defaultdict(list)
    for m in meta:
        c = cell_of_meta(m)
        if c and float(m["votes_avg"] or 0) >= VOTES and 2.0 <= float(m["duration"] or 0) <= 10.0 and clean(m["text_tr"]) not in bad_txt:
            by[(int(m["reciter_id"]), c)].append(m)
    real = {}
    for rid in sorted(tgt_ids) + distract:
        s = f"aihub_{rid:04d}"
        for c in EMO7:
            ms = sorted([m for m in by[(rid, c)] if f"71349_{m['file_id']}" not in eref and f"71349_{m['file_id']}" != ref_utt.get(s)], key=lambda m: m["file_id"])
            random.Random(f"1986|refsim|{s}|{c}").shuffle(ms)
            if ms:
                real[f"{s}|{c}"] = [dict(utt=f"71349_{m['file_id']}", audio=str(RAW / m["audio_path"]), text=m["text_tr"], intensity=m["intensity"],
                                         votes=float(m["votes_avg"])) for m in ms[:NMAX]]
    rows542 = defaultdict(list)
    for line in open(E542):
        r = json.loads(line)
        rows542[r["speaker"]].append(r)
    for s, v in P["spk"].items():
        if v["src"] != "542":
            continue
        rt = clean(v["ref_text"])
        rs = sorted([r for r in rows542.get(s, []) if clean(r["text"]) not in rt and clean(r["text"]) not in bad_txt], key=lambda r: r["utt"])
        random.Random(f"1986|refsim|{s}|Neutrality").shuffle(rs)
        real[f"{s}|Neutrality"] = [dict(utt=r["utt"], audio=r["audio"], text=r["text"]) for r in rs[:NMAX]]
    info = dict(targets_unseen8=sorted(un), distract=[f"aihub_{r:04d}" for r in distract],
                gender={**{f"aihub_{r:04d}": rid_g[r] for r in list(tgt_ids) + distract}, **{s: v["gender"] for s, v in P["spk"].items()}},
                rule=f"71349 투표 ≥ {VOTES} · 2~10초 · 강도 1 → A · 2/3 → B · 무감정 강도 0 · 칸마다 최대 {NMAX} · seed 1986")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(info=info, real=real), open(OUT / "real_plan.json", "w"), ensure_ascii=False, indent=1)
    print("후보 화자", len(distract), "· 성별", dict(sorted({g: sum(info["gender"][d] == g for d in info["distract"]) for g in ("male", "female")}.items())))
    for s in sorted(un) + [s for s, v in P["spk"].items() if v["src"] == "542"]:
        print(s, " · ".join(f"{KO[c]} {len(real.get(f'{s}|{c}', []))}" for c in EMO7))
    print("후보 칸별 화자 수(≥3개)", {KO[c]: sum(len(real.get(f"{d}|{c}", [])) >= 3 for d in info["distract"]) for c in EMO7})


def cmd_emb(a):
    import librosa
    sys.path.insert(0, str(H / "q3tts_koft"))
    from q3ko.eval.sv import load_embedder
    R = jl(OUT / "real_plan.json")["real"]
    paths = sorted({x["audio"] for v in R.values() for x in v})
    f = OUT / "emb_real.npz"
    F = {"w": load_embedder("wavlm_large", device="cuda:0"), "e": load_embedder("eres2net", device="cuda:0")}
    out = {"w": [], "e": []}
    for i, p in enumerate(paths):
        y = librosa.load(p, sr=16000)[0]
        for k in F:
            out[k].append(np.asarray(F[k](y, 16000), dtype=np.float32))
        if (i + 1) % 300 == 0:
            print(i + 1, "/", len(paths), flush=True)
    np.savez(f, paths=np.array(paths), w=np.stack(out["w"]), e=np.stack(out["e"]))
    print("✅", len(paths))


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def load_model(tag):
    z = np.load(VS / f"emb_{tag}.npz")
    ids = [str(i) for i in z["ids"]]
    out = {k: defaultdict(list) for k in ("w", "e")}
    for k in out:
        for i, v in zip(ids, z[k]):
            for c in EMO7:
                if f"_{c}_" in i:
                    s = i.split(f"_{c}_")[0]
                    out[k][(s, c)].append(unit(v))
    return out


def cmd_analyze(a):
    P = jl(D / "plan.json")
    RP = jl(OUT / "real_plan.json")
    R, info = RP["real"], RP["info"]
    z = np.load(OUT / "emb_real.npz")
    RE = {k: {str(p): unit(v) for p, v in zip(z["paths"], z[k])} for k in ("w", "e")}
    ref_z = np.load(VS / "emb_ref.npz") if (VS / "emb_ref.npz").exists() else None
    tags = TAGS + [t for t in a.extra_tags if t not in TAGS]
    M = {t: load_model(t) for t in tags}
    un = info["targets_unseen8"]
    all16 = list(P["spk"])
    gender = info["gender"]
    rv = lambda k, s, c: [RE[k][x["audio"]] for x in R.get(f"{s}|{c}", [])]  # noqa: E731
    res = dict(info=dict(tags=tags, unseen8=un, n_real={f"{s}|{c}": len(R.get(f"{s}|{c}", [])) for s in all16 for c in EMO7}, distract=info["distract"]))

    # (가) 같은 칸 비교(unseen8)
    ga = {}
    for k in ("w", "e"):
        ceil, top_real = {}, {}
        for c in EMO7:
            vals, hits = [], []
            for s in un:
                X = rv(k, s, c)
                if len(X) < 3:
                    continue
                X = np.stack(X)
                for i in range(len(X)):
                    rest = np.delete(X, i, 0)
                    vals.append(float((rest @ X[i]).mean()))
                    # 사람 1위: 하나 뺀 나머지 중심 vs 후보 중심
                    cand = {d: unit(np.mean(rv(k, d, c), 0)) for d in info["distract"] if gender[d] == gender[s] and len(rv(k, d, c)) >= 3}
                    cand[s] = unit(rest.mean(0))
                    sims = {d: float(v @ X[i]) for d, v in cand.items()}
                    hits.append(max(sims, key=sims.get) == s)
            ceil[c] = dict(sim=float(np.mean(vals)) if vals else None, top1=100 * float(np.mean(hits)) if hits else None, n=len(vals))
        mod = {}
        for t in tags:
            row = {}
            for c in EMO7:
                byN = {n: [] for n in NS}
                neu = []
                hits, n_cand = [], []
                for s in un:
                    X = rv(k, s, c)
                    if len(X) < 3:
                        continue
                    O = M[t][k].get((s, c), [])
                    for o in O:
                        for n in NS:
                            if len(X) >= n:
                                byN[n].append(float((np.stack(X[:n]) @ o).mean()))
                        N_ = rv(k, s, "Neutrality")
                        if N_:
                            neu.append(float((np.stack(N_) @ o).mean()))
                        cand = {d: unit(np.mean(rv(k, d, c), 0)) for d in info["distract"] if gender[d] == gender[s] and len(rv(k, d, c)) >= 3}
                        cand[s] = unit(np.mean(X, 0))
                        sims = {d: float(v @ o) for d, v in cand.items()}
                        hits.append(max(sims, key=sims.get) == s)
                        n_cand.append(len(cand))
                if not byN[1]:
                    continue
                row[c] = dict(**{f"N{n}": float(np.mean(v)) if v else None for n, v in byN.items()},
                              vs_neutral_real=float(np.mean(neu)) if neu else None,
                              top1=100 * float(np.mean(hits)), n_cand=float(np.mean(n_cand)), n=len(byN[1]))
                ce = ceil[c]["sim"]
                row[c]["pct_ceiling"] = 100 * row[c]["N10"] / ce if (ce and row[c]["N10"]) else None
            mod[t] = row
        ga[k] = dict(ceiling=ceil, model=mod)
    res["ga"] = ga

    # (나) REF 개수 효과(16명 · 무감정 진짜 녹음 N개) — 모든 칸 출력 평균 · 칸별
    na = {}
    for k in ("w", "e"):
        out = {}
        for t in tags:
            byN = {n: [] for n in NS}
            byNc = {c: {n: [] for n in NS} for c in EMO7}
            for s in all16:
                X = rv(k, s, "Neutrality")
                for c in EMO7:
                    for o in M[t][k].get((s, c), []):
                        for n in NS:
                            if len(X) >= n:
                                v = float((np.stack(X[:n]) @ o).mean())
                                byN[n].append(v)
                                byNc[c][n].append(v)
            out[t] = dict(all={n: float(np.mean(v)) for n, v in byN.items() if v}, cell={c: {n: float(np.mean(v)) for n, v in d.items() if v} for c, d in byNc.items()},
                          n_spk={n: sum(len(rv(k, s, "Neutrality")) >= n for s in all16) for n in NS})
        rank = {n: sorted(tags, key=lambda t: -out[t]["all"].get(n, -9)) for n in NS}
        # REF 하나(지금 잣대 t_sim 과 같은 꼴: d_eval sv 는 real 2개 평균 · 여기서는 N=1 이 가장 가까움)
        na[k] = dict(model=out, rank=rank)
    res["na"] = na

    # (다) 칸 쌍 표
    def pair_tab(get, spks):
        T = {}
        for c1, c2 in itertools.combinations_with_replacement(EMO7, 2):
            vals = []
            for s in spks:
                A, B = get(s, c1), get(s, c2)
                if c1 == c2:
                    if len(A) >= 2:
                        vals += [float(A[i] @ A[j]) for i in range(len(A)) for j in range(i + 1, len(A))]
                elif A and B:
                    vals += [float(x @ y) for x in A for y in B]
            T[f"{c1}|{c2}"] = float(np.mean(vals)) if vals else None
        return T
    da = {}
    for k in ("w", "e"):
        real_spk = {s: pair_tab(lambda s_, c: rv(k, s_, c), [s]) for s in un}
        real_tab = pair_tab(lambda s_, c: rv(k, s_, c), un)
        mt = {t: pair_tab(lambda s_, c: M[t][k].get((s_, c), []), un) for t in tags}
        mt16 = {t: pair_tab(lambda s_, c: M[t][k].get((s_, c), []), all16) for t in tags}
        mspk = {t: {s: pair_tab(lambda s_, c: M[t][k].get((s_, c), []), [s]) for s in all16} for t in (MAIN,)}
        # 사람 대비 낙차(모델 − 사람) — 같은 쌍을 진짜 녹음으로 잴 수 있는 unseen8 화자만
        gap = {}
        for t in tags:
            g = {}
            for pk in real_tab:
                vs = [(mspk_ := None)]  # noqa: F841
                diffs = []
                for s in un:
                    r_ = real_spk[s].get(pk)
                    m_ = pair_tab(lambda s_, c: M[t][k].get((s_, c), []), [s]).get(pk) if t != MAIN else mspk[MAIN][s].get(pk)
                    if r_ is not None and m_ is not None:
                        diffs.append(m_ - r_)
                g[pk] = dict(gap=float(np.mean(diffs)) if diffs else None, n_spk=len(diffs))
            gap[t] = g
        # 화자 · 칸 쌍 단위 낙차(MAIN · 청음 줄 고르기)
        rows = []
        for s in un:
            for pk, r_ in real_spk[s].items():
                m_ = mspk[MAIN][s].get(pk)
                c1, c2 = pk.split("|")
                if r_ is not None and m_ is not None and c1 != c2:
                    rows.append(dict(spk=s, pair=pk, model=m_, real=r_, gap=m_ - r_))
        rows.sort(key=lambda r: r["gap"])
        da[k] = dict(real=real_tab, real_by_spk=real_spk, model_unseen8=mt, model_16=mt16, model_by_spk_main=mspk[MAIN], gap=gap, worst_rows=rows[:20])
    res["da"] = da

    # (라) 기쁨A 후보 섞기
    ra = {}
    ser = {t: jl(D / f"ser_{t}.json") for t in HAPPY_CAND}
    sv = {t: jl(D / f"sv_{t}.json") for t in HAPPY_CAND}
    for k in ("w", "e"):
        out = {}
        for t in HAPPY_CAND:
            vals, per = [], {}
            for s in all16:
                H_ = M[t][k].get((s, "Happy-A"), [])
                O6 = [v for c in EMO7 if c != "Happy-A" for v in M[MAIN][k].get((s, c), [])]
                if H_ and O6:
                    v = float(np.mean([h @ o for h in H_ for o in O6]))
                    vals.append(v)
                    per[s] = v
            out[t] = dict(mean=float(np.mean(vals)), per_spk=per)
        for t in HAPPY_CAND:
            out[t]["drop"] = out[t]["mean"] - out[MAIN]["mean"]
            out[t]["drop_min_spk"] = min(out[t]["per_spk"][s] - out[MAIN]["per_spk"][s] for s in out[t]["per_spk"])
        ra[k] = out
    hap = {}
    for t in HAPPY_CAND:
        xs = [x for x in P["items"] if x["emo"] == "Happy-A"]
        hap[t] = dict(e2v_happy=100 * float(np.mean([ser[t][x["id"]]["top"] == "happy" for x in xs])),
                      p_happy=float(np.mean([ser[t][x["id"]]["happy"] for x in xs])),
                      arousal=float(np.mean([ser[t][x["id"]]["arousal"] for x in xs])),
                      t_sim=float(np.mean([sv[t][x["id"]]["t_sim"] for x in xs])))
    res["ra"] = dict(sim=ra, happy=hap)
    json.dump(res, open(OUT / "analysis.json", "w"), ensure_ascii=False, indent=1)
    show(res)


def show(res):
    for k in ("w", "e"):
        g = res["ga"][k]
        print(f"\n=== (가) 같은 칸 [{k}] 사람 천장:", {KO[c]: (None if v["sim"] is None else round(v["sim"], 3), v["top1"] and round(v["top1"]), v["n"]) for c, v in g["ceiling"].items()})
        for t, row in g["model"].items():
            print(f"  {t:22s}", " · ".join(f"{KO[c]} {v['N10'] if v['N10'] is None else round(v['N10'], 3)}({v['pct_ceiling'] and round(v['pct_ceiling'])}% · 1위 {v['top1']:.0f})" for c, v in row.items()))
        n = res["na"][k]
        print(f"=== (나) [{k}] 순위:", {N: r[:4] for N, r in n["rank"].items()})
        for t in n["model"]:
            print(f"  {t:22s}", {N: round(v, 3) for N, v in n["model"][t]["all"].items()})
        d = res["da"][k]
        print(f"=== (다) [{k}] {MAIN} 사람 대비 낙차 큰 쌍:", sorted(((round(v["gap"], 3), p, v["n_spk"]) for p, v in d["gap"][MAIN].items() if v["gap"] is not None))[:8])
        print("   화자 · 쌍:", [(r["spk"], r["pair"], round(r["gap"], 3)) for r in d["worst_rows"][:8]])
        r = res["ra"]["sim"][k]
        print(f"=== (라) [{k}]", {t: (round(v["mean"], 3), round(v["drop"], 3), round(v["drop_min_spk"], 3)) for t, v in r.items()})
    print("기쁨A e2v", {t: round(v["e2v_happy"]) for t, v in res["ra"]["happy"].items()})


# ───────────────────────────── §4-1 작은 청음 페이지(최대 12줄)
SPK6 = ["a542_2214_G2A2E7_JCE", "a542_2108_G2A5E7_KKH", "a542_1837_G1A2E7_KHG", "aihub_0152", "aihub_0115", "aihub_0156"]   # listen_pages.SPK6
HAPPY_TEXT = "바다가 너무 예뻐서 기분이 좋았어."
PAGE_CSS = """body{font-family:sans-serif;max-width:1800px;margin:16px auto;line-height:1.45;padding:0 10px}table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccc;padding:5px;vertical-align:top;font-size:.88em}th{background:#f3f3f3}.n{color:#666;font-size:.8em}audio{width:180px}
button{margin:1px;font-size:.8em}button.on{background:#1565c0;color:#fff}#bar{position:sticky;top:0;background:#fff;padding:6px;border-bottom:1px solid #ccc;z-index:9}
body.hidenum .num{display:none}textarea{width:96%;font-size:.85em}"""
PAGE_JS = r"""<script>
const K=PAGE.key;let S=JSON.parse(localStorage.getItem(K)||'{}');
function save(){localStorage.setItem(K,JSON.stringify(S));cnt()}
function cnt(){document.getElementById('cnt').textContent=Object.keys(S).filter(k=>!k.endsWith('|memo')).length}
function draw(){document.querySelectorAll('.b').forEach(d=>d.querySelectorAll('button').forEach(b=>b.classList.toggle('on',S[d.dataset.id]===b.dataset.v)));
 document.querySelectorAll('textarea[data-id]').forEach(t=>t.value=S[t.dataset.id]||'');cnt()}
document.querySelectorAll('.b button').forEach(b=>b.onclick=()=>{const id=b.parentElement.dataset.id;if(S[id]===b.dataset.v)delete S[id];else S[id]=b.dataset.v;save();draw()});
document.querySelectorAll('textarea[data-id]').forEach(t=>t.oninput=()=>{if(t.value.trim())S[t.dataset.id]=t.value;else delete S[t.dataset.id];save()});
function q(s){s=(s==null?'':String(s));return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
function dl(){const L=[['줄','화자','종류','열','질문','답','메모'].join(',')];
 for(const [rid,r] of Object.entries(PAGE.rows)){
  for(const [c,qs] of Object.entries(r.cols)){for(const qq of qs){const v=S[rid+'|'+c+'|'+qq];if(v)L.push([rid,r.spk,r.kind,c,qq,v,''].map(q).join(','))}}
  if(S[rid+'|memo'])L.push([rid,r.spk,r.kind,'','','',S[rid+'|memo']].map(q).join(','))}
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\ufeff'+L.join('\n')],{type:'text/csv'}));a.download=PAGE.csv;a.click()}
draw();
</script>"""


def worst_pairs(res, n=6):
    """(다) 화자 · 칸 쌍 낙차(모델 − 사람)를 두 잣대 평균으로 → 가장 큰 n개"""
    g = defaultdict(dict)
    for k in ("w", "e"):
        for s in res["info"]["unseen8"]:
            rs = res["da"][k]["real_by_spk"][s]
            ms = res["da"][k]["model_by_spk_main"][s]
            for pk, r_ in rs.items():
                c1, c2 = pk.split("|")
                if c1 != c2 and r_ is not None and ms.get(pk) is not None:
                    g[(s, pk)][k] = (ms[pk] - r_, ms[pk], r_)
    rows = [dict(spk=s, pair=pk, gap=float(np.mean([v[0] for v in d.values()])), gw=d["w"][0], ge=d["e"][0], mw=d["w"][1], rw=d["w"][2])
            for (s, pk), d in g.items() if len(d) == 2]
    rows.sort(key=lambda r: r["gap"])
    return rows[:n]


def cmd_page(a):
    import html
    import librosa
    import soundfile as sf
    E = html.escape
    TL = H / "tmp_listen/vox_koft"
    O = TL / "ref_check"
    (O / "real").mkdir(parents=True, exist_ok=True)
    rep = jl(OUT / "repro.json")
    assert rep["all_ok"], "재현 실패 — 페이지 안 만듦"
    res = jl(OUT / "analysis.json")
    P = jl(D / "plan.json")
    R = jl(OUT / "real_plan.json")["real"]
    sv = {t: jl(D / f"sv_{t}.json") for t in HAPPY_CAND}
    ser6 = {t: jl(TL / f"listen_set6/ser_{t}.json") for t in HAPPY_CAND}
    A = lambda p: f'<audio controls preload="none" src="{p}"></audio>'  # noqa: E731

    def btn(i, opts):
        return "<div class=b data-id='%s'>%s</div>" % (E(i), "".join(f"<button data-v='{o}'>{o}</button>" for o in opts))
    rows, h = {}, [f"<!doctype html><meta charset=utf-8><title>REF 확인 · 12줄</title><style>{PAGE_CSS}</style>",
                   "<div id=bar>판정 <span id=cnt>0</span>개 · <button onclick='dl()'>CSV 내려받기</button> · "
                   "<label><input type=checkbox onchange=\"document.body.classList.toggle('hidenum',this.checked)\"> 숫자 숨기기(귀 먼저)</label> · "
                   "<button onclick=\"if(confirm('판정 · 메모를 모두 지울까요?')){localStorage.removeItem(PAGE.key);location.reload()}\">판정 지우기</button></div>",
                   "<h1>🎧 vox_koft REF 확인 — 12줄</h1>",
                   "<p><b>REF 와 조금 달라도, 같은 줄의 소리끼리 같은 사람이면 '같음'</b></p>",
                   f"<p class=n>만든 때 {__import__('time').strftime('%m/%d %H:%M', __import__('time').gmtime(__import__('time').time() + 9 * 3600))} KST · "
                   "보고서 docs/reports/20261109_ref_similarity_by_cell.md · 음원은 11/08 B · C 페이지 · D 재측정 음원 재사용(노트북 엔진 재현 확인 "
                   f"{rep['n_ok']}/{rep['n']})</p>",
                   "<h2>줄 1~6 · 기쁨A 후보 — NUX2sN s3000 다른 칸과 섞어도 같은 사람인가</h2>",
                   f"<p class=n>문장 \"{E(HAPPY_TEXT)}\" · 기준 소리 = NUX2sN s3000 중립(같은 사람 기준) · 후보마다 두 가지: "
                   "<b>중립과 같은 사람?</b>(같음 · 애매 · 다름) · <b>기쁨?</b>(OK · 약함 · 틀림)</p>",
                   "<table><tr><th>화자</th><th>REF</th><th>기준: NUX2sN s3000 중립</th>"
                   + "".join(f"<th>기쁨A 후보 {E(t)}</th>" for t in HAPPY_CAND) + "<th>메모</th></tr>"]
    for i, s in enumerate(SPK6, 1):
        rid = f"h{i}"
        v = P["spk"][s]
        cells = []
        for t in HAPPY_CAND:
            q = ser6[t].get(f"{s}_Happy-A_new", {})
            num = f"<div class='n num'>e2v {q.get('top', '?')} · happy {q.get('happy', 0):.2f}</div>" if q else ""
            cells.append(f"<td>{A(f'../listen_set6/wav/{t}/{s}_Happy-A_new.wav')}{num}{btn(f'{rid}|{t}|같은사람', ['같음', '애매', '다름'])}{btn(f'{rid}|{t}|기쁨', ['OK', '약함', '틀림'])}</td>")
        rows[rid] = dict(spk=s, kind="기쁨A 후보", cols={t: ["같은사람", "기쁨"] for t in HAPPY_CAND})
        h.append(f"<tr><td><b>{i}. {E(s)}</b><div class=n>{'여' if v['gender'] == 'female' else '남'} · {v['src']}</div></td>"
                 f"<td>{A(f'../d_eval/ref/{s}_ref.wav')}</td><td>{A(f'../d_eval/wav/{MAIN}/{s}_Neutrality_0.wav')}<div class=n>중립(D 재측정 문장)</div></td>"
                 + "".join(cells) + f"<td><textarea rows=3 data-id='{rid}|memo'></textarea></td></tr>")
    h.append("</table>")
    W = worst_pairs(res)
    h.append("<h2>줄 7~12 · 숫자상 \"사람보다 많이 멀어진\" 칸 쌍(NUX2sN s3000)</h2>"
             "<p class=n>같은 사람의 두 칸 출력이, 그 사람의 진짜 두 칸 녹음끼리보다 얼마나 더 먼가(WavLM · ERes2Net 평균 낙차) — 가장 큰 6개. "
             "진짜 녹음은 말투 · 문장 · 녹음 환경이 다릅니다(참고용). 버튼: <b>두 출력이 같은 사람?</b></p>"
             "<table><tr><th>화자 · 칸 쌍</th><th>REF</th><th>출력 칸 1</th><th>출력 칸 2</th><th>진짜 칸 1</th><th>진짜 칸 2</th><th>같은 사람?</th><th>메모</th></tr>")
    for j, r in enumerate(W, 7):
        rid = f"p{j}"
        c1, c2 = r["pair"].split("|")
        sp = r["spk"]
        reals = []
        for c in (c1, c2):
            x = R[f"{sp}|{c}"][0]
            d = O / "real" / f"{x['utt']}.wav"
            if not d.exists():
                sf.write(d, librosa.load(x["audio"], sr=24000)[0], 24000)
            reals.append(f"<td>{A('real/' + d.name)}<div class=n>{KO[c]} · {E(x['text'])}</div></td>")
        outs = [f"<td>{A(f'../d_eval/wav/{MAIN}/{sp}_{c}_0.wav')}<div class=n>{KO[c]}</div></td>" for c in (c1, c2)]
        rows[rid] = dict(spk=sp, kind=f"칸 쌍 {KO[c1]}-{KO[c2]}", cols={"출력": ["같은사람"]})
        h.append(f"<tr><td><b>{j}. {E(sp)}</b><div class=n>{KO[c1]} ↔ {KO[c2]}</div><div class='n num'>낙차 {r['gap']:+.3f}(W {r['gw']:+.3f} · E {r['ge']:+.3f})<br>"
                 f"출력끼리 W {r['mw']:.3f} · 진짜끼리 W {r['rw']:.3f}</div></td><td>{A(f'../d_eval/ref/{sp}_ref.wav')}</td>"
                 + "".join(outs) + "".join(reals)
                 + f"<td>{btn(f'{rid}|출력|같은사람', ['같음', '애매', '다름'])}</td><td><textarea rows=3 data-id='{rid}|memo'></textarea></td></tr>")
    h.append("</table>")
    page = dict(key="refchk_judg", rows=rows, csv="vox_koft_ref_check.csv")
    h.append("<script>const PAGE=" + json.dumps(page, ensure_ascii=False) + ";</script>" + PAGE_JS)
    (O / "index.html").write_text("\n".join(h), encoding="utf-8")
    json.dump(dict(worst=W, rows=rows), open(O / "page_meta.json", "w"), ensure_ascii=False, indent=1)
    print("✅", O / "index.html", "· 줄", len(rows), [(r["spk"], r["pair"], round(r["gap"], 3)) for r in W])


def cmd_repro(a):
    """페이지 음원이 노트북 엔진(voxko.listen.Listener)과 같은 소리인지 — 모델마다 1개(기쁨A 새 문장 · MAIN 은 중립 문항도)"""
    import soundfile as sf
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import listen_pages as LP
    from voxko.listen import Listener
    TL = H / "tmp_listen/vox_koft"
    St, P = jl(TL / "listen_set6/set.json"), jl(D / "plan.json")
    norm = St["norm"][HAPPY_TEXT]
    out = []
    for k, t in enumerate(HAPPY_CAND):
        short, st = t.rsplit("_s", 1)
        L = Listener(LP.RUN[short][0])
        s = SPK6[k % len(SPK6)]
        jobs = [(norm, "Happy-A", TL / f"listen_set6/wav/{t}/{s}_Happy-A_new.wav", s)]
        if t == MAIN:
            x = next(y for y in P["items"] if y["id"] == f"{SPK6[-1]}_Neutrality_0")
            jobs.append((x["text"], "Neutrality", D / f"wav/{t}/{x['id']}.wav", SPK6[-1]))
        for text, emo, wp, sp in jobs:
            w, sr, _ = L.synth(int(st), text, emotion=emo, ref={"ref": P["spk"][sp]["ref"]}, normalize=False)
            w1 = sf.read(wp)[0]
            ok = bool(len(w) == len(w1) and float(np.abs(w - w1).max()) < 1e-3)
            out.append(dict(tag=t, file=str(wp), same=ok, len=[len(w), len(w1)]))
            print(t, wp.name, "같은 소리:", ok, flush=True)
        del L
        torch.cuda.empty_cache()
    json.dump(dict(items=out, n=len(out), n_ok=sum(x["same"] for x in out), all_ok=all(x["same"] for x in out)), open(OUT / "repro.json", "w"), indent=1)
    print(f"REPRO {sum(x['same'] for x in out)}/{len(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pick", "emb", "analyze", "show", "repro", "page"])
    ap.add_argument("--extra_tags", nargs="*", default=[])
    a = ap.parse_args()
    if a.cmd == "show":
        show(jl(OUT / "analysis.json"))
    else:
        {"pick": cmd_pick, "emb": cmd_emb, "analyze": cmd_analyze, "repro": cmd_repro, "page": cmd_page}[a.cmd](a)
