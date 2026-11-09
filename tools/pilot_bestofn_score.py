"""tools/pilot_bestofn.py 의 채점 · 고르기 · 페이지 단계 (2026-11-09) — 설명은 pilot_bestofn.py 맨 위.

고르기 규칙(11/06 V3 바탕 · 문턱은 이 파일 상수 · 근거는 보고서 docs/reports/20261109_pilot_bestofn.md)
  1. 버림: CER > 10% · d_eval 이상(빈 합성 · 안 끝남 · 긴 무음 · 클리핑 · 끝 잘림 · SQUIM 이 원본(base) 하위 5% 보다 낮음)
           · 갈라짐 후보 = aper(말소리 큰 프레임 중 주기성 낮은 비율 · tools/crackle_probe.py)가 그 칸 후보 전체의 90% 분위 이상
             (사용자 메모 210개로 시험: aper AUC 0.77 · SQUIM PESQ 는 0.36 이라 갈라짐을 못 잡음)
  2. 감정 있음: e2v 1위가 그 칸 감정 · 또는 arousal 이 진짜 녹음(A7N val) 같은 칸 10~90% 범위 안
           · B칸은 arousal 이 그 문항 후보(seed 4개) 중앙값 이상(세기 지키기 · 11/09 청음 교훈)
           · 중립은 e2v 1위 neutral 또는 arousal 이 진짜 중립 범위 안
  3. 같은 사람: own = "그 화자의 고른 중립 출력 평균"과의 코사인(WavLM · ERes2Net 평균) ≥ 진짜 녹음 하위 20% 문턱(칸 계열별)
           · 고른 중립: 중립 후보 중 1 · 2 를 통과한 것들의 평균(첫 기준)에 가장 가까운 것을 문장마다 1개 → 그 10개 평균이 기준
  4. 남은 것 중 own 이 가장 높은 1개
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_bestofn as PB  # noqa: E402

DE, PL, D, H = PB.DE, PB.PL, PB.D, PB.H
TAGS = ["main", "hA_b", "hA_c", "hA_d", "hA_e"]
CER_MAX, APER_Q = 0.10, 90
KO = {"Neutrality": "중립", "Happy-A": "기쁨A", "Happy-B": "기쁨B", "Sad-A": "슬픔A", "Sad-B": "슬픔B", "Angry-A": "분노A", "Angry-B": "분노B"}


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def jl(p):
    return json.load(open(p))


# ───────────────────────────── emb (env q3tts) — WavLM-large · ERes2Net
def emb(a):
    import librosa
    sys.path.insert(0, str(H / "q3tts_koft"))
    from q3ko.eval.sv import load_embedder
    F = {"w": load_embedder("wavlm_large", device="cuda:0"), "e": load_embedder("eres2net", device="cuda:0")}
    for t in (a.tag.split(",") if a.tag != "all" else TAGS):
        P = PB.plan_of(t)
        f = PL / f"emb_{t}.npz"
        ids = [x["id"] for x in P["items"]]
        if f.exists() and list(np.load(f)["ids"]) == ids:
            print(t, "이미 있음")
            continue
        out = {k: [] for k in F}
        for n, i in enumerate(ids):
            w = librosa.load(str(PL / "wav" / t / f"{i}.wav"), sr=16000)[0]
            for k in F:
                out[k].append(F[k](w, 16000))
            if n % 1000 == 0:
                print(t, n, flush=True)
        np.savez(f, ids=np.array(ids), **{k: np.stack(v) for k, v in out.items()})
        print("emb", t, len(ids), flush=True)
    f = PL / "emb_ref.npz"                                                         # 화자 REF · 진짜 녹음(무감정) — 참고
    if not f.exists():
        P = PB.plan_of("main")
        keys, out = [], {k: [] for k in F}
        for s, v in P["spk"].items():
            for kind, p in [("REF", v["ref"])] + [("real", r) for r in v["real"]]:
                w = librosa.load(p, sr=16000)[0]
                keys.append(f"{s}|{kind}|{Path(p).stem}")
                for k in F:
                    out[k].append(F[k](w, 16000))
        np.savez(f, ids=np.array(keys), **{k: np.stack(v) for k, v in out.items()})


# ───────────────────────────── ser (env q3tts_ser) — e2v · arousal · CER (조각)
def _cer(r, h):
    d = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(h)] / max(1, len(r))


def ser(a):
    sys.path.insert(0, str(H / "q3tts_koft"))
    from ser.models import dim_infer, e2v_infer, load_16k, load_dim, load_e2v, load_whisper
    e2v, dim, wh = load_e2v(), load_dim(), load_whisper()
    clean = lambda t: re.sub(r"[^가-힣a-zA-Z0-9]", "", t)  # noqa: E731
    for t in (a.tag.split(",") if a.tag != "all" else TAGS):
        P = PB.plan_of(t)
        f = PL / f"ser_{t}_{a.shard.replace('/', 'of')}.json"
        res = jl(f) if f.exists() else {}
        for n, x in enumerate(PB.shard(P["items"], a.shard)):
            if x["id"] in res:
                continue
            w = load_16k(str(PL / "wav" / t / f"{x['id']}.wav"))
            labs, sc = e2v_infer(e2v, w)
            d = dict(zip(labs, sc))
            dm = dim_infer(dim, w)
            hyp = wh.transcribe(w, language="ko", temperature=0.0, condition_on_previous_text=False)["text"]
            res[x["id"]] = dict(top=labs[int(np.argmax(sc))], happy=d.get("happy", 0), sad=d.get("sad", 0), angry=d.get("angry", 0),
                                neutral=d.get("neutral", 0), arousal=float(dm.get("arousal", np.nan)), valence=float(dm.get("valence", np.nan)),
                                cer=_cer(clean(x["text"]), clean(hyp)), asr=hyp.strip())
            if n % 200 == 0:
                json.dump(res, open(f, "w"), ensure_ascii=False)
                print(t, a.shard, n, flush=True)
        json.dump(res, open(f, "w"), ensure_ascii=False)
        print("ser", t, a.shard, len(res), flush=True)


def real_aro(a):
    """진짜 녹음(A7N val · 7칸) e2v · arousal → 칸별 arousal 10~90% 범위."""
    sys.path.insert(0, str(H / "q3tts_koft"))
    from ser.models import dim_infer, e2v_infer, load_16k, load_dim, load_e2v
    e2v, dim = load_e2v(), load_dim()
    out = {}
    for r in DE.rd(DE.VD / "val.jsonl"):
        w = load_16k(r["audio"])
        labs, sc = e2v_infer(e2v, w)
        out[r["utt"]] = dict(emo=r["emotion"], top=labs[int(np.argmax(sc))], arousal=float(dim_infer(dim, w).get("arousal", np.nan)))
    json.dump(out, open(PL / "real_arousal.json", "w"), ensure_ascii=False, indent=1)
    print("진짜 녹음 arousal", len(out))


# ───────────────────────────── quality (env voxcpm) — d_eval 이상 + SQUIM + 갈라짐 후보(aper …)
def quality(a):
    import soundfile as sf
    import torch
    import torchaudio
    from multiprocessing import Pool
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    from voxko.eval.metrics import tail_metrics
    import crackle_probe as CP
    m = SQUIM_OBJECTIVE.get_model().to("cuda").eval()
    cap = PB.MAX_LEN * 0.16
    for t in (a.tag.split(",") if a.tag != "all" else TAGS):
        P = PB.plan_of(t)
        f = PL / f"quality_{t}_{a.shard.replace('/', 'of')}.json"
        res = jl(f) if f.exists() else {}
        xs = [x for x in PB.shard(P["items"], a.shard) if x["id"] not in res]
        paths = [str(PL / "wav" / t / f"{x['id']}.wav") for x in xs]
        with Pool(8) as pool:
            FT = pool.map(CP.feats, paths, chunksize=8)
        for x, p, ft in zip(xs, paths, FT):
            y, sr = sf.read(p, dtype="float32")
            dur = len(y) / sr
            with torch.no_grad():
                z = torchaudio.functional.resample(torch.from_numpy(y), sr, 16000)[None].cuda()
                stoi, pesq, sisdr = (float(v) for v in m(z)) if z.shape[-1] > 1600 else (float("nan"),) * 3
            hop = sr // 100
            n = len(y) // hop
            peak = float(np.abs(y).max()) + 1e-12
            fr = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-12 if n else np.zeros(1)
            idx = np.where(20 * np.log10(fr / peak) > -40)[0]
            gap = float(np.diff(idx).max() - 1) * 0.01 if len(idx) > 1 else 0.0
            voiced = (idx[-1] - idx[0] + 1) * 0.01 if len(idx) else 0.0
            syl = len(re.findall(r"[가-힣]", x["text"]))
            tm = tail_metrics(y, sr)
            res[x["id"]] = dict(dur=round(dur, 3), stoi=stoi, pesq=pesq, sisdr=sisdr, max_gap=round(max(gap, 0.0), 2),
                                clip=float((np.abs(y) >= 0.999).mean()), cut=bool(tm["cut_suspect"]), tail_sil_ms=tm["tail_sil_ms"],
                                empty=bool(dur < DE.EMPTY_SEC or peak < 1e-3), hit_max=bool(dur >= cap - 0.5),
                                sps=round(syl / voiced, 2) if voiced > 0 else 0.0, syl=syl, peak=peak, **ft)
        json.dump(res, open(f, "w"))
        print("quality", t, a.shard, len(res), flush=True)


def merge(a=None):
    for t in TAGS:
        for kind in ("ser", "quality"):
            parts = sorted(PL.glob(f"{kind}_{t}_*of*.json"))
            if not parts:
                continue
            res = {}
            for p in parts:
                res.update(jl(p))
            n = len(PB.plan_of(t)["items"])
            json.dump(res, open(PL / f"{kind}_{t}.json", "w"), ensure_ascii=False)
            print(kind, t, f"{len(res)}/{n}")


# ───────────────────────────── select
def real_thr():
    """진짜 녹음(d_eval vspk emb_real · unseen8 16명): 감정 녹음 ↔ 그 화자 무감정 평균 · 무감정 ↔ (나머지 무감정 + REF) 평균 → 칸 계열별 p20."""
    z = np.load(D / "vspk/emb_real.npz")
    meta = json.loads(str(z["meta"]))
    sims = defaultdict(list)
    for s in sorted({m["spk"] for m in meta}):
        idx = [j for j, m in enumerate(meta) if m["spk"] == s]
        nidx = [j for j in idx if meta[j]["grp"] == "Neutrality"]
        ridx = [j for j in idx if meta[j]["grp"] == "REF"]
        for j in idx:
            g = meta[j]["grp"]
            if g in ("Happy", "Sad", "Angry"):
                sims[g].append(float(np.mean([unit(z[k][j]) @ unit(np.mean([unit(z[k][n]) for n in nidx], 0)) for k in ("w", "e")])))
            elif g == "Neutrality":
                oth = [n for n in nidx if n != j] + ridx
                sims[g].append(float(np.mean([unit(z[k][j]) @ unit(np.mean([unit(z[k][n]) for n in oth], 0)) for k in ("w", "e")])))
    return {g: dict(p10=float(np.percentile(v, 10)), p20=float(np.percentile(v, 20)), mean=float(np.mean(v)), n=len(v)) for g, v in sims.items()}


def aro_range():
    by = defaultdict(list)
    for v in jl(PL / "real_arousal.json").values():
        by[v["emo"]].append(v["arousal"])
    return {e: (float(np.percentile(v, 10)), float(np.percentile(v, 90)), len(v)) for e, v in by.items()}


def load(tag):
    """문항별 지표 + 1 · 2 단계 판정."""
    P = PB.plan_of(tag)
    z = np.load(PL / f"emb_{tag}.npz")
    zi = {i: j for j, i in enumerate(z["ids"])}
    sr, qa = jl(PL / f"ser_{tag}.json"), jl(PL / f"quality_{tag}.json")
    return P, z, zi, sr, qa


def judge(P, z, zi, sr, qa, thr_sq, ar, aper_thr):
    out = {}
    for x in P["items"]:
        i, e = x["id"], x["emo"]
        q, a = sr[i], qa[i]
        o = DE.odd(a, thr_sq)
        if a["aper"] >= aper_thr[e]:
            o = o + ["갈라짐 후보"]
        lo, hi, _ = ar[e]
        hit = q["top"] == DE.E2V[DE.base(e)]
        v = dict(x, top=q["top"], hit=hit, arousal=q["arousal"], cer=q["cer"], asr=q["asr"], odd=o, aper=a["aper"], pesq=a["pesq"], dur=a["dur"],
                 p_emo=q.get(DE.E2V[DE.base(e)], np.nan) if DE.base(e) != "Neutrality" else q.get("neutral", np.nan),
                 vec={k: unit(z[k][zi[i]]) for k in ("w", "e")})
        v["emo_in"] = hit or (lo <= q["arousal"] <= hi)
        out[i] = v
    by = defaultdict(list)
    for v in out.values():
        by[v["base_id"]].append(v)
    for vs in by.values():
        med = float(np.median([u["arousal"] for u in vs]))
        for v in vs:
            v["aro_med"] = v["arousal"] >= med
            r = []
            if v["cer"] > CER_MAX:
                r.append("CER > 10%")
            r += v["odd"]
            v["drop1"] = r
            r2 = []
            if not v["emo_in"]:
                r2.append("감정 없음")
            if v["emo"].endswith("-B") and not v["aro_med"]:
                r2.append("B칸 arousal 중앙값 미만")
            v["drop2"] = r2
            v["ok12"] = not r and not r2
    return out, by


def own(v, cen):
    return float(np.mean([v["vec"][k] @ cen[k] for k in ("w", "e")]))


def pick(by, T, cen_of, loo_neutral=None):
    """문항(base_id)마다 3 · 4 단계."""
    res = {}
    for b, vs in by.items():
        vs = sorted(vs, key=lambda v: v["seed"])
        s = vs[0]["tgt"]
        g = DE.base(vs[0]["emo"])
        cen = cen_of(s, b)
        for v in vs:
            v["own"] = own(v, cen)
            v["same"] = v["own"] >= T[g]["p20"]
            v["reasons"] = v["drop1"] + v["drop2"] + ([] if v["same"] else ["같은 사람 문턱 미만"])
        c = [v for v in vs if v["ok12"] and v["same"]]
        ch = max(c, key=lambda v: v["own"]) if c else None
        res[b] = dict(all=vs, s86=vs[0], chosen=ch, grp=g, emo=vs[0]["emo"], tgt=s)
    return res


def select(a=None):
    thr_sq = DE._sq_thr()
    T, ar = real_thr(), aro_range()
    M = load("main")
    allq = {**M[4]}
    aper_thr = {e: float(np.percentile([allq[x["id"]]["aper"] for x in M[0]["items"] if x["emo"] == e], APER_Q)) for e in PB.CELLS}
    items, by = judge(*M, thr_sq, ar, aper_thr)
    # 고른 중립: 1 · 2 통과 중립 평균(첫 기준)에 가장 가까운 것 → 문장마다 1개 → 10개 평균
    neu = defaultdict(list)
    for v in items.values():
        if v["emo"] == "Neutrality" and v["ok12"]:
            neu[v["tgt"]].append(v)
    first = {s: {k: unit(np.mean([v["vec"][k] for v in vs], 0)) for k in ("w", "e")} for s, vs in neu.items()}
    npick = defaultdict(dict)
    for s, vs in neu.items():
        for v in vs:
            o = own(v, first[s])
            if v["base_id"] not in npick[s] or o > npick[s][v["base_id"]][0]:
                npick[s][v["base_id"]] = (o, v)
    ncen = {s: {k: unit(np.mean([v["vec"][k] for _, v in d.values()], 0)) for k in ("w", "e")} for s, d in npick.items()}

    def cen_of(s, b):
        if b in npick[s]:                                                          # 중립 문항은 자기 빼고(leave-one-out)
            oth = [v for bb, (_, v) in npick[s].items() if bb != b]
            return {k: unit(np.mean([v["vec"][k] for v in oth], 0)) for k in ("w", "e")}
        return ncen[s]
    R = pick(by, T, cen_of)
    out = dict(real_thr=T, aro_range=ar, squim_thr=thr_sq, aper_thr=aper_thr, cer_max=CER_MAX, seeds=PB.SEEDS,
               n_neutral_ok={s: len(d) for s, d in npick.items()})
    out["cells"] = cell_table(R, items)
    # 기쁨A 후보
    ha = {"main": {b: r for b, r in R.items() if r["emo"] == "Happy-A"}}
    hitems = {"main": {i: v for i, v in items.items() if v["emo"] == "Happy-A"}}
    for t in TAGS[1:]:
        if not (PL / f"emb_{t}.npz").exists():
            continue
        it, bb = judge(*load(t), thr_sq, ar, aper_thr)
        ha[t] = pick(bb, T, cen_of)
        hitems[t] = it
    others = defaultdict(lambda: {k: [] for k in ("w", "e")})                     # NUX2sN 의 다른 6칸(고른 것)
    for r in R.values():
        if r["chosen"] and r["emo"] not in ("Happy-A",):
            for k in ("w", "e"):
                others[r["tgt"]][k].append(r["chosen"]["vec"][k])
    out["happyA"] = {t: ha_table(rr, hitems[t], others) for t, rr in ha.items()}
    out["f0"] = {}
    json.dump(out, open(PL / "report.json", "w"), ensure_ascii=False, indent=1)
    write_picked(R, PL / "picked_main.jsonl")
    for t, rr in ha.items():
        if t != "main":
            write_picked(rr, PL / f"picked_{t}.jsonl")
    print_tables(out)


def cell_table(R, items):
    out = {}
    for e in PB.CELLS + ["전체"]:
        rs = [r for r in R.values() if e == "전체" or r["emo"] == e]
        vs = [v for r in rs for v in r["all"]]
        ch = [r["chosen"] for r in rs if r["chosen"]]
        s86 = [r["s86"] for r in rs]
        first = Counter()
        for v in vs:
            if v["reasons"]:
                first[v["reasons"][0]] += 1
        allr = Counter(x for v in vs for x in set(v["reasons"]))
        m = lambda xs, k: float(np.mean([v[k] for v in xs])) if xs else float("nan")  # noqa: E731
        out[e] = dict(n_items=len(rs), n_cand=len(vs), yield_=100 * len(ch) / len(rs), cand_pass=100 * np.mean([not v["reasons"] for v in vs]),
                      first_reason={k: 100 * c / len(vs) for k, c in first.most_common()}, any_reason={k: 100 * c / len(vs) for k, c in allr.most_common()},
                      before=dict(own=m(s86, "own"), hit=100 * m(s86, "hit"), arousal=m(s86, "arousal"), cer=100 * m(s86, "cer"),
                                  aper=m(s86, "aper"), odd=100 * float(np.mean([bool(v["drop1"]) for v in s86])), same=100 * m(s86, "same")),
                      after=dict(own=m(ch, "own"), hit=100 * m(ch, "hit"), arousal=m(ch, "arousal"), cer=100 * m(ch, "cer"),
                                 aper=m(ch, "aper"), odd=0.0, same=100.0))
    # 화자별 수율(7칸 모두 채운 문장 수)
    return out


def ha_table(R, items, others):
    ch = [r["chosen"] for r in R.values() if r["chosen"]]
    vs = [v for r in R.values() for v in r["all"]]
    o6 = []
    for v in ch:
        O = others[v["tgt"]]
        o6.append(float(np.mean([np.mean(np.stack(O[k]) @ v["vec"][k]) for k in ("w", "e")])))
    m = lambda xs, k: float(np.mean([v[k] for v in xs])) if xs else float("nan")  # noqa: E731
    first = Counter(v["reasons"][0] for v in vs if v["reasons"])
    return dict(n=len(R), yield_=100 * len(ch) / len(R), all_hit=100 * m(vs, "hit"), all_arousal=m(vs, "arousal"), all_p=m(vs, "p_emo"),
                hit=100 * m(ch, "hit"), arousal=m(ch, "arousal"), p_happy=m(ch, "p_emo"), own=m(ch, "own"), all_own=m(vs, "own"),
                other6=float(np.mean(o6)) if o6 else float("nan"), cer=100 * m(ch, "cer"),
                first_reason={k: 100 * c / len(vs) for k, c in first.most_common()})


def write_picked(R, f):
    with open(f, "w") as fo:
        for b, r in sorted(R.items()):
            c = r["chosen"]
            rows = [dict(seed=v["seed"], own=round(v["own"], 4), hit=v["hit"], top=v["top"], arousal=round(v["arousal"], 3), cer=round(v["cer"], 3),
                         aper=round(v["aper"], 3), reasons=v["reasons"]) for v in r["all"]]
            fo.write(json.dumps(dict(base_id=b, spk=r["tgt"], cell=r["emo"], text=r["all"][0]["text"], picked_seed=c["seed"] if c else None,
                                     picked_id=c["id"] if c else None, score=dict(own=round(c["own"], 4), hit=c["hit"], arousal=round(c["arousal"], 3),
                                                                                    cer=round(c["cer"], 3)) if c else None,
                                     candidates=rows), ensure_ascii=False) + "\n")


def print_tables(out):
    print("문턱: 같은 사람 p20", {g: round(v["p20"], 3) for g, v in out["real_thr"].items()}, "· aper", {KO[e]: round(v, 3) for e, v in out["aper_thr"].items()})
    for e, c in out["cells"].items():
        b, f = c["before"], c["after"]
        top = list(c["first_reason"].items())[:3]
        print(f"{KO.get(e, e):4s} 수율 {c['yield_']:.0f}% · 후보 통과 {c['cand_pass']:.0f}% · own {b['own']:.3f}→{f['own']:.3f} · 맞힘 {b['hit']:.0f}→{f['hit']:.0f}% · "
              f"aro {b['arousal']:.2f}→{f['arousal']:.2f} · CER {b['cer']:.1f}→{f['cer']:.1f} · 같은 사람 {b['same']:.0f}% · 버림 " + " · ".join(f"{k} {v:.0f}%" for k, v in top))
    for t, h in out["happyA"].items():
        print(f"{PB.HA_LAB[t]:24s} 수율 {h['yield_']:.0f}% · 전체 맞힘 {h['all_hit']:.0f}% aro {h['all_arousal']:.2f} · 고른 것 맞힘 {h['hit']:.0f}% aro {h['arousal']:.2f} "
              f"p {h['p_happy']:.2f} · own {h['own']:.3f} · 다른 6칸 {h['other6']:.3f}")


def f0(a=None):
    """기쁨A 후보 F0(REF 대비 반음 · 고른 것) — tools/angryB_pitch.py 규칙(bcell_compare 와 같음)."""
    from multiprocessing import Pool
    import angryB_pitch as AB
    P = PB.plan_of("main")
    R = jl(PL / "report.json")
    with Pool(32) as pool:
        ref = dict(zip(P["spk"], [v for _, v, _ in pool.map(AB.f0_med, [(v["ref"], 60.0, 800.0) for v in P["spk"].values()])]))
        out = {}
        for t in TAGS:
            f = PL / ("picked_main.jsonl" if t == "main" else f"picked_{t}.jsonl")
            rows = [json.loads(l) for l in open(f)]
            rows = [r for r in rows if r["cell"] == "Happy-A" and r["picked_id"]]
            jobs = [(str(PL / "wav" / t / f"{r['picked_id']}.wav"), ref[r["spk"]] / 2, ref[r["spk"]] * 2 ** (20 / 12)) for r in rows]
            res = pool.map(AB.f0_med, jobs)
            st = [12 * np.log2(v / ref[r["spk"]]) for r, (_, v, vs) in zip(rows, res) if vs >= 0.3 and np.isfinite(v)]
            out[t] = dict(n=len(st), st_med=float(np.median(st)), st_mean=float(np.mean(st)))
            print(PB.HA_LAB[t], out[t])
    R["f0"] = out
    json.dump(R, open(PL / "report.json", "w"), ensure_ascii=False, indent=1)


# ───────────────────────────── page — 파일럿 청음(최대 20줄)
PCSS = """body{font-family:sans-serif;max-width:1700px;margin:16px auto;line-height:1.45;padding:0 10px}table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccc;padding:5px;vertical-align:top;font-size:.88em}th{background:#f3f3f3}.n{color:#666;font-size:.8em}audio{width:200px}
button{margin:1px;font-size:.8em}button.on{background:#1565c0;color:#fff}#bar{position:sticky;top:0;background:#fff;padding:6px;border-bottom:1px solid #ccc;z-index:9}
body.hidenum .num{display:none}textarea{width:96%;font-size:.85em}.q{font-size:.78em;color:#333;margin-top:3px}.guide{background:#fff8e1;padding:8px;border:1px solid #f0d58a}"""
PJS = r"""<script>
const K=PAGE.key;let S=JSON.parse(localStorage.getItem(K)||'{}');
document.querySelectorAll('audio').forEach(a=>a.volume=0.35);
function save(){localStorage.setItem(K,JSON.stringify(S));cnt()}
function cnt(){document.getElementById('cnt').textContent=Object.keys(S).filter(k=>!k.endsWith('|memo')).length}
function draw(){document.querySelectorAll('.b').forEach(d=>d.querySelectorAll('button').forEach(b=>b.classList.toggle('on',S[d.dataset.id]===b.dataset.v)));
 document.querySelectorAll('textarea[data-id]').forEach(t=>t.value=S[t.dataset.id]||'');cnt()}
document.querySelectorAll('.b button').forEach(b=>b.onclick=()=>{const id=b.parentElement.dataset.id;if(S[id]===b.dataset.v)delete S[id];else S[id]=b.dataset.v;save();draw()});
document.querySelectorAll('textarea[data-id]').forEach(t=>t.oninput=()=>{if(t.value.trim())S[t.dataset.id]=t.value;else delete S[t.dataset.id];save()});
function q(s){s=(s==null?'':String(s));return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
function dl(){const L=[['페이지','줄','화자','칸','문장','열','seed','목소리','감정','음질','메모'].join(',')];
 for(const [rid,r] of Object.entries(PAGE.rows)){
  for(const [c,sd] of r.cols){const v=S[rid+'|'+c+'|v'],e=S[rid+'|'+c+'|e'],a=S[rid+'|'+c+'|a'];if(v||e||a)L.push([PAGE.name,r.no,r.spk,r.cell,r.text,c,sd,v||'',e||'',a||'',''].map(q).join(','))}
  if(S[rid+'|memo'])L.push([PAGE.name,r.no,r.spk,r.cell,r.text,'','','','','',S[rid+'|memo']].map(q).join(','))}
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['﻿'+L.join('\n')],{type:'text/csv'}));a.download=PAGE.csv;a.click()}
draw();
</script>"""


def _rows_plan(R, HA):
    """줄 1~14: 칸마다 2줄(여 1 · 남 1 · 14줄 모두 다른 화자) · 고른 seed ≠ 1986 인 문항 우선 · 줄 15~20: 기쁨A 후보 6명(여 3 · 남 3 · 앞 14명과 다른 화자 먼저)."""
    import random
    P = PB.plan_of("main")
    rng = random.Random(f"{DE.SEED}|pilot_page")
    fem = sorted(s for s, v in P["spk"].items() if v["gender"] == "female")
    mal = sorted(s for s, v in P["spk"].items() if v["gender"] == "male")
    rng.shuffle(fem)
    rng.shuffle(mal)
    rows, used = [], set()
    for ci, e in enumerate(PB.CELLS):
        for pool in (fem, mal):
            for s in pool:
                if s in used:
                    continue
                c = [r for b, r in sorted(R.items()) if r["tgt"] == s and r["emo"] == e and r["chosen"]]
                if not c:
                    continue
                pref = [r for r in c if r["chosen"]["seed"] != DE.SEED] or c
                rows.append(("main", rng.choice(pref)))
                used.add(s)
                break
    left = lambda pool: [s for s in pool if s not in used] + [s for s in pool if s in used]  # noqa: E731
    for pool in (fem, mal):
        n = 0
        for s in left(pool):
            bs = [b for b, r in sorted(R.items()) if r["tgt"] == s and r["emo"] == "Happy-A" and all(HA[t].get(b, {}).get("chosen") for t in HA)]
            if bs and n < 3:
                rows.append(("hA", rng.choice(bs)))
                n += 1
    return rows


def page(a=None):
    import html
    import shutil
    E = html.escape
    thr_sq = DE._sq_thr()
    T, ar = real_thr(), aro_range()
    M = load("main")
    aper_thr = jl(PL / "report.json")["aper_thr"]
    items, by = judge(*M, thr_sq, ar, aper_thr)
    # select() 와 같은 고른 중립 · 기준(다시 계산 — 같은 함수)
    neu = defaultdict(list)
    for v in items.values():
        if v["emo"] == "Neutrality" and v["ok12"]:
            neu[v["tgt"]].append(v)
    first = {s: {k: unit(np.mean([v["vec"][k] for v in vs], 0)) for k in ("w", "e")} for s, vs in neu.items()}
    npick = defaultdict(dict)
    for s, vs in neu.items():
        for v in vs:
            o = own(v, first[s])
            if v["base_id"] not in npick[s] or o > npick[s][v["base_id"]][0]:
                npick[s][v["base_id"]] = (o, v)
    ncen = {s: {k: unit(np.mean([v["vec"][k] for _, v in d.values()], 0)) for k in ("w", "e")} for s, d in npick.items()}

    def cen_of(s, b):
        if b in npick[s]:
            oth = [v for bb, (_, v) in npick[s].items() if bb != b]
            return {k: unit(np.mean([v["vec"][k] for v in oth], 0)) for k in ("w", "e")}
        return ncen[s]
    R = pick(by, T, cen_of)
    HA = {"main": {b: r for b, r in R.items() if r["emo"] == "Happy-A"}}
    for t in TAGS[1:]:
        it, bb = judge(*load(t), thr_sq, ar, aper_thr)
        HA[t] = pick(bb, T, cen_of)
    # 고른 중립 대표 1개 = 기준(평균)에 가장 가까운 고른 중립
    rep_neu = {s: max((v for _, v in d.values()), key=lambda v: own(v, ncen[s])) for s, d in npick.items()}
    P = M[0]
    (PL / "ref").mkdir(exist_ok=True)
    for s, v in P["spk"].items():
        if not (PL / "ref" / f"{s}.wav").exists():
            shutil.copy(v["ref"], PL / "ref" / f"{s}.wav")
    A = lambda p: f'<audio controls preload="none" src="{p}"></audio>'  # noqa: E731

    def btn(i, opts):
        return "<div class=b data-id='%s'>%s</div>" % (E(i), "".join(f"<button data-v='{o}'>{o}</button>" for o in opts))

    def num(v):
        return (f"<div class='n num'>seed {v['seed']} · own {v['own']:.3f} · e2v {v['top']} · aro {v['arousal']:.2f} · CER {100 * v['cer']:.0f}% · aper {v['aper']:.2f}"
                + (f" · ⚠️ {' · '.join(v['reasons'])}" if v["reasons"] else "") + "</div>")
    rows = _rows_plan(R, HA)
    meta = {}
    h = [f"<!doctype html><meta charset=utf-8><title>파일럿 · 여러 번 만들어 고르기</title><style>{PCSS}</style>",
         "<div id=bar>판정 <span id=cnt>0</span>개 · <button onclick='dl()'>CSV 내려받기</button> · "
         "<label><input type=checkbox onchange=\"document.body.classList.toggle('hidenum',this.checked)\"> 숫자 숨기기(귀 먼저)</label> · "
         "<button onclick=\"if(confirm('판정 · 메모를 모두 지울까요?')){localStorage.removeItem(PAGE.key);location.reload()}\">판정 지우기</button></div>",
         "<h2>파일럿 데이터셋 — NUX2sN s3000 · 여러 번(seed 4개) 만들어 고르기</h2>",
         "<p class=guide><b>REF 와 조금 달라도, 같은 줄의 소리끼리 같은 사람이면 '같음'</b> · 볼륨은 낮게 시작합니다(35%).</p>",
         "<p class=n>줄 1~14: 칸마다 2줄(여 1 · 남 1). 열 = REF(그 화자 무감정 진짜 녹음) · 고른 중립(이 화자의 기준 목소리) · <b>고르기 전(seed 1986)</b> · <b>고른 것</b>. "
         "두 열을 비교해 고르기가 귀로도 효과가 있는지 봅니다. 줄 15~20: 기쁨A 후보 5개(같은 화자 · 같은 문장 · 각자 고른 것). "
         "11/09 ref_check 페이지 줄 1~6(기쁨A 후보)과 겹치므로, 이 페이지를 들으면 <b>그 페이지는 건너뛰어도 됩니다</b>.</p>",
         "<table><tr><th style='width:40px'>줄</th><th style='width:170px'>화자 · 칸 · 문장</th><th>REF</th><th>고른 중립</th><th>판정할 소리</th><th style='width:180px'>메모</th></tr>"]
    for no, (kind, x) in enumerate(rows, 1):
        if kind == "main":
            r = x
            s, e = r["tgt"], r["emo"]
            rid = f"r{no:02d}"
            cols = []
            judged = []
            for lab, v in (("고르기 전(seed 1986)", r["s86"]), ("고른 것", r["chosen"])):
                cols.append([lab, v["seed"]])
                judged.append(f"<div style='display:inline-block;vertical-align:top;margin-right:14px'><b>{lab}</b><br>{A('wav/main/' + v['id'] + '.wav')}{num(v)}"
                              f"<div class=q>중립과 같은 사람?</div>{btn(f'{rid}|{lab}|v', ['같음', '애매', '다름'])}"
                              f"<div class=q>감정</div>{btn(f'{rid}|{lab}|e', ['OK', '약함', '틀림'])}"
                              f"<div class=q>음질</div>{btn(f'{rid}|{lab}|a', ['깨끗', '갈라짐'])}</div>")
            same = " <span class=n>(고르기 전 = 고른 것 · 같은 seed)</span>" if r["s86"]["id"] == r["chosen"]["id"] else ""
            text = r["s86"]["text"]
        else:
            b = x
            r = R[b]
            s, e = r["tgt"], "Happy-A"
            rid = f"r{no:02d}"
            cols, judged = [], []
            for t in TAGS:
                v = HA[t][b]["chosen"]
                lab = PB.HA_LAB[t]
                cols.append([lab, v["seed"]])
                judged.append(f"<div style='display:inline-block;vertical-align:top;margin-right:10px'><b>{E(lab)}</b><br>{A(f'wav/{t}/' + v['id'] + '.wav')}{num(v)}"
                              f"<div class=q>중립과 같은 사람?</div>{btn(f'{rid}|{lab}|v', ['같음', '애매', '다름'])}"
                              f"<div class=q>기쁨?</div>{btn(f'{rid}|{lab}|e', ['OK', '약함', '틀림'])}</div>")
            same = ""
            text = r["all"][0]["text"]
        g = "여" if P["spk"][s]["gender"] == "female" else "남"
        meta[rid] = dict(no=no, spk=s, cell=KO[e] + ("(후보)" if kind == "hA" else ""), text=text, cols=cols)
        nv = rep_neu[s]
        h.append(f"<tr><td>{no}</td><td><b>{E(s)}</b><div class=n>{g} · 542</div><b>{KO[e]}</b>{' · 기쁨A 후보' if kind == 'hA' else ''}<div>{E(text)}</div>{same}</td>"
                 f"<td>{A('ref/' + s + '.wav')}</td><td>{A('wav/main/' + nv['id'] + '.wav')}<div class='n num'>{E(nv['text'][:18])}…</div></td>"
                 f"<td>{''.join(judged)}</td><td><textarea rows=4 data-id='{rid}|memo'></textarea></td></tr>")
    h.append("</table><p class=n>입력: \"(v1 지시문)문장\" + 그 화자 무감정 REF · cfg 2.0 · timesteps 10 · seed 1986~1989 중 고름. (나)만 추론 문구 "
             f"\"{E(PB.HA_BRIGHT)}\". 숫자: own = 고른 중립 평균과의 목소리 유사도(WavLM · ERes2Net 평균) · aro = arousal · aper = 갈라짐 지표. 숫자는 참고용 — 판정은 귀로.</p>")
    h.append("<script>const PAGE=" + json.dumps(dict(key="pilot_judg", name="pilot", rows=meta, csv="vox_koft_pilot.csv"), ensure_ascii=False) + ";</script>" + PJS)
    (PL / "index.html").write_text("\n".join(h), encoding="utf-8")
    json.dump(dict(rows=[(k, (x["base_id"] if k == "main" else x) if isinstance(x, str) else x["all"][0]["base_id"]) for k, x in rows]),
              open(PL / "page_meta.json", "w"), ensure_ascii=False, indent=1)
    print("페이지", len(rows), "줄 →", PL / "index.html")
