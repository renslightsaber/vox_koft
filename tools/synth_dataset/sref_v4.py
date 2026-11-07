"""SREF v4 후보 풀 · 고르기 · 만들기(2026-11-07 · q3tts_koft 세션 · 사용자 결정 "후보 풀 + UTMOSv2 시험")
흐름(run_sref_v4.sh 가 차례로 부름)
  · 후보 폴더 v4/new/<cand>/ 마다 sentences.jsonl · wav24/ — VoxCPM2 seed 1987 · 1988 · 1989(vox_s1~3) · SREF2 s600(sref2_s0 · s1 · seed 1986 · 1987)
    SREF2 후보는 영어만 조각 → English 태그 · 코드스위치/영어 의문문 → Korean 태그(통째) · 인용 · 조사 조각은 SREF2 후보 없음
  · 채점(기존 도구 그대로): sref_synth.py score(목소리 · 끝 잘림 · 속도) · asr(CER) · quality_check.py(SQUIM · 무음 · 클리핑) + 여기 asr_en(영어 고정 CER · logprob)
  · select: 후보마다 **기존 필터 전부** 적용 → 통과 후보 중 순위(영어만 = UTMOSv2 가 쓸 만하면 UTMOS, 아니면 목소리) · SREF2 출신은 묶음마다 절반 이하
  · build: v3.1 − 사용자 '빼야 함' 4 + 고른 것 → SREF/v4/sref_{train,val}.jsonl(+ lang 칸) · dataset_info.json
단계
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/q3tts/bin/python sref_v4.py sref2 --cand sref2_s0 --seed 1986
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/q3tts_ser/bin/python sref_v4.py asr_en --cand vox_s1
  ~/miniforge3/envs/q3tts/bin/python sref_v4.py select   ·   sref_v4.py build
"""
import argparse, json, re, sys, time, random
from collections import Counter
from pathlib import Path
H = Path.home() / "workspaces"; S = H / "preprocessed/q3tts_koft/SREF"; V31 = S / "v3_1"; V4 = S / "v4"; NEW = V4 / "new"
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
SEED = 1986
USER_DROP = {"v3eo_0247", "v3eo_0580", "v3eo_0206", "v3eo_0029"}
# 기존 필터(바꾸지 않음): sref_synth.py · make_sref_v3.py · sref_v3_en_rescore.py 와 같은 값
CER_MAX, CER_MAX_CS, SIM_MIN, SIM_MIN_EN, CPS = 0.10, 0.20, 0.60, 0.50, (2.5, 9.0)
CER_EN_MAX, LP_MIN, GAP_MAX, CLIP_MAX = 0.15, -0.4691, 0.8, 1e-4          # LP_MIN = v3.1 의 ASR 확신도 하위 20% 문턱(고정)
SREF2_CAP = 0.5
VOX = ["vox_s1", "vox_s2", "vox_s3"]; SR2 = ["sref2_s0", "sref2_s1"]
SREF2_KINDS = {"en_only", "codeswitch", "en_q_embed"}


def wj(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def lang_of(text):
    """서빙 2094420 처럼 단어 다수결: 한글이 든 단어 vs 영어 글자만 있는 단어."""
    ko = en = 0
    for w in text.split():
        if re.search(r"[가-힣]", w): ko += 1
        elif re.search(r"[A-Za-z]", w): en += 1
    return "English" if en > ko else "Korean"


def sref2(a):
    """SREF2_xvec_NSOSFT_01 step 600 으로 후보 합성(노트북 엔진 SrefListener 그대로 · 문장은 이미 정규화됨 → normalize=False)."""
    sys.path.insert(0, str(H / "q3tts_koft/notebooks")); sys.path.insert(0, str(H / "q3tts_koft"))
    import numpy as np, soundfile as sf
    from utils import listen_sref as ls
    od = NEW / a.cand; (od / "wav24").mkdir(parents=True, exist_ok=True)
    Sx = jl(od / "sentences.jsonl")
    if a.shard:
        i, n = map(int, a.shard.split("/")); Sx = Sx[i::n]
    L = ls.SrefListener("SREF2_xvec_NSOSFT_01", follow_data=False); L.use_step(600)
    L.synth("준비 합성입니다.", seed=a.seed)                       # 첫 합성 준비
    meta, t0 = {}, time.time()
    for s in Sx:
        p = od / "wav24" / f"{s['id']}.wav"
        if p.exists():
            continue
        lang = "English" if s["kind"] == "en_only" else "Korean"
        w, _ = L.synth(s["text"], seed=a.seed, normalize=False, language=lang)
        sf.write(p, w, 24000); meta[s["id"]] = dict(dur=len(w) / 24000, seed=a.seed, language=lang)
        L.WAVS.clear()
    w2, _ = L.synth(Sx[0]["text"], seed=a.seed, normalize=False, language="English" if Sx[0]["kind"] == "en_only" else "Korean")
    w1 = sf.read(od / "wav24" / f"{Sx[0]['id']}.wav")[0]
    rep = len(w1) == len(w2) and float(np.abs(w1 - w2).max()) < 1e-3
    json.dump(dict(meta=meta, repro=rep, model="SREF2_xvec_NSOSFT_01/step_600", sec=time.time() - t0), open(od / f"synth_sref2_{(a.shard or 'all').replace('/', 'of')}.json", "w"), indent=1)
    print("SREF2 합성", len(meta), "재현", rep)


def asr_en(a):
    """영어만 조각: whisper 영어 고정 · 평균 log 확률(sref_v3_en_rescore.asr 와 같음)."""
    sys.path.insert(0, str(H / "q3tts_koft"))
    from ser.models import load_whisper, load_16k
    od = NEW / a.cand; wh = load_whisper(); out = {}
    for s in jl(od / "sentences.jsonl"):
        p = od / "wav24" / f"{s['id']}.wav"
        if s["kind"] != "en_only" or not p.exists():
            continue
        r = wh.transcribe(load_16k(str(p)), language="en", temperature=0.0, condition_on_previous_text=False)
        segs = r.get("segments") or []
        lp = sum(x["avg_logprob"] * (x["end"] - x["start"]) for x in segs) / max(1e-6, sum(x["end"] - x["start"] for x in segs)) if segs else -9.0
        out[s["id"]] = dict(asr_en=r["text"].strip(), avg_logprob=round(lp, 4))
    json.dump(out, open(od / "asr_en.json", "w"), ensure_ascii=False, indent=1); print("ASR(en)", a.cand, len(out))


def _norm_en(t):
    from ko_tts_norm.en_numbers import read_token
    t = t.lower().replace("’", "'"); toks = []
    for w in re.findall(r"[a-z']+|\$?\d[\d:.,]*", t):
        toks.append(((read_token(w.rstrip(".,")) or w).lower()) if re.match(r"\$?\d", w) else w)
    return re.sub(r"[^a-z]", "", " ".join(toks).replace("'", ""))


def _cer(r, h):
    d = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(h)] / max(1, len(r))


def judge(s, c, r, q, e, thr):
    """후보 하나에 기존 필터 전부. 반환 = 떨어진 이유 목록(빈 목록 = 통과)."""
    why = []
    en_only = s["kind"] == "en_only"
    if c["cut"]: why.append("끝 잘림")
    if c["sim"] < (SIM_MIN_EN if en_only else SIM_MIN): why.append(f"목소리 {c['sim']:.2f}")
    if en_only:
        if e is None: why.append("영어 ASR 없음")
        else:
            ce = _cer(_norm_en(s["text"]), _norm_en(e["asr_en"]))
            if ce > CER_EN_MAX: why.append(f"영어 CER {ce:.2f}")
            if e["avg_logprob"] < LP_MIN: why.append(f"ASR 확신도 {e['avg_logprob']:.2f}")
    else:
        cmax = CER_MAX_CS if (s["kind"] == "codeswitch" or s.get("en")) else CER_MAX
        if r["cer_ko"] > cmax: why.append(f"CER {r['cer_ko']:.2f}")
        if not CPS[0] <= c["cps"] <= CPS[1]: why.append(f"속도 {c['cps']:.1f}")
    why += [k for k in ("stoi", "pesq", "sisdr") if q[k] < thr[k]]
    if q["max_gap"] >= GAP_MAX: why.append("긴 무음")
    if q["clip"] > CLIP_MAX: why.append("클리핑")
    return why


def select(a):
    thr = json.load(open(S / "v1/quality_summary.json"))["thr"]
    ut = json.load(open(NEW / "utmos_check.json")) if (NEW / "utmos_check.json").exists() else {}
    use_utmos = bool(ut.get("use"))
    U = {}
    for cand in VOX + SR2:
        f = NEW / cand / "utmos.json"
        if use_utmos and f.exists():
            U[cand] = json.load(open(f))
    sent = {s["id"]: s for s in jl(NEW / "sentences.jsonl")}
    C = {}
    for cand in VOX + SR2:
        od = NEW / cand
        if not (od / "score.json").exists():
            continue
        sc, ar, qq = json.load(open(od / "score.json")), json.load(open(od / "asr.json")), json.load(open(od / "quality.json"))
        ae = json.load(open(od / "asr_en.json")) if (od / "asr_en.json").exists() else {}
        for i in sc:
            if i not in ar or i not in qq:
                continue
            s = sent[i]; why = judge(s, sc[i], ar[i], qq[i], ae.get(i), thr)
            C.setdefault(i, []).append(dict(cand=cand, ok=not why, why=why, sim=sc[i]["sim"], dur=sc[i]["dur"], cer=ar[i]["cer_ko"],
                                            asr=ar[i]["asr"], lp=(ae.get(i) or {}).get("avg_logprob"), asr_en=(ae.get(i) or {}).get("asr_en"),
                                            utmos=U.get(cand, {}).get(i), squim={k: round(qq[i][k], 3) for k in ("stoi", "pesq", "sisdr")},
                                            max_gap=qq[i]["max_gap"]))
    def key(s, c):
        if s["kind"] == "en_only" and use_utmos and c["utmos"] is not None:
            return (c["utmos"], c["sim"])
        return (c["sim"], c["lp"] or 0.0)
    pick = {}
    for i, cs in C.items():
        ok = sorted([c for c in cs if c["ok"]], key=lambda c: key(sent[i], c), reverse=True)
        if ok:
            pick[i] = ok
    # SREF2 출신 ≤ 절반(묶음마다): 넘치면 VoxCPM2 대안이 있는 문장 중 손해가 작은 것부터 바꿈
    chosen = {i: v[0] for i, v in pick.items()}
    for k in sorted({sent[i]["kind"] for i in chosen}):
        ids = [i for i in chosen if sent[i]["kind"] == k]
        over = sum(chosen[i]["cand"].startswith("sref2") for i in ids) - int(SREF2_CAP * len(ids))
        if over <= 0:
            continue
        sw = []
        for i in ids:
            if chosen[i]["cand"].startswith("sref2"):
                alt = [c for c in pick[i] if c["cand"].startswith("vox")]
                if alt:
                    k0, k1 = key(sent[i], chosen[i]), key(sent[i], alt[0]); sw.append((k0[0] - k1[0], i, alt[0]))
        for _, i, alt in sorted(sw)[:over]:
            chosen[i] = alt
        # 대안이 없어 여전히 넘치면 SREF2 출신 중 점수 낮은 것부터 뺌
        ids = [i for i in chosen if sent[i]["kind"] == k]
        s2 = sorted([i for i in ids if chosen[i]["cand"].startswith("sref2")], key=lambda i: key(sent[i], chosen[i]))
        over = len(s2) - int(SREF2_CAP * len(ids))
        for i in s2[:max(0, over)]:
            chosen.pop(i)
    summ = dict(sentences=len(sent), with_candidates=len(C), passed_any=len(pick), chosen=len(chosen), use_utmos=use_utmos, utmos_check=ut,
                by_kind={k: dict(total=sum(s["kind"] == k for s in sent.values()), chosen=sum(sent[i]["kind"] == k for i in chosen),
                                 sref2=sum(sent[i]["kind"] == k and chosen[i]["cand"].startswith("sref2") for i in chosen))
                         for k in sorted({s["kind"] for s in sent.values()})},
                by_cand=dict(Counter(c["cand"] for c in chosen.values())),
                cand_pass_rate={cand: round(sum(c["ok"] for v in C.values() for c in v if c["cand"] == cand) /
                                            max(1, sum(1 for v in C.values() for c in v if c["cand"] == cand)), 3) for cand in VOX + SR2},
                reasons=dict(Counter(w.split()[0] for v in C.values() for c in v for w in c["why"])),
                rule=dict(sim_min=SIM_MIN, sim_min_en=SIM_MIN_EN, cer=CER_MAX, cer_cs=CER_MAX_CS, cer_en=CER_EN_MAX, logprob_min=LP_MIN,
                          cps=CPS, squim=thr, gap=GAP_MAX, clip=CLIP_MAX, sref2_cap=SREF2_CAP))
    json.dump(dict(summary=summ, chosen=chosen, candidates=C), open(NEW / "select.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


def build(a):
    R = json.load(open(NEW / "select.json")); chosen, summ = R["chosen"], R["summary"]
    sent = {s["id"]: s for s in jl(NEW / "sentences.jsonl")}
    svc = json.loads((H / "reference_audios/qwen3tts/set2_trim03_rms20/female_ment_natural_24k_trim03_rms20.json").read_text(encoding="utf-8"))
    SERVICE = str(H / "reference_audios/qwen3tts/set2_trim03_rms20/female_ment_natural_24k_trim03_rms20.wav")
    tr31, va31 = jl(V31 / "sref_train.jsonl"), jl(V31 / "sref_val.jsonl")
    tr31 = [r for r in tr31 if r["utt"] not in USER_DROP]; va31 = [r for r in va31 if r["utt"] not in USER_DROP]
    dropped = USER_DROP - {r["utt"] for r in jl(V31 / "sref_train.jsonl") + jl(V31 / "sref_val.jsonl")}
    new = []
    for i, c in sorted(chosen.items()):
        s = sent[i]; src = "sref2_s600" if c["cand"].startswith("sref2") else "voxcpm2"
        new.append(dict(audio=str(NEW / c["cand"] / "wav24" / f"{i}.wav"), text=s["text"], ref_audio=SERVICE, ref_text=svc["normalized_text"],
                        speaker="service", language="Auto", duration=round(c["dur"], 3), kind=s["kind"], utt=i,
                        src_corpus=("sref2_selfgen" if src == "sref2_s600" else "voxcpm2_sref"), raw_text=s["raw"], en_type=s.get("en_type"),
                        profile=s.get("profile"), normalizer_commit=s.get("normalizer_commit"), cand=c["cand"], gen_source=src,
                        retry_of=s.get("retry_of"), squim=c["squim"], max_gap=c["max_gap"], src_version="v4new",
                        score=dict(sim=round(c["sim"], 4), cer=round(c["cer"], 4), asr=c["asr"], logprob=c["lp"], asr_en=c["asr_en"], utmos=c["utmos"])))
    # 검증: v3.1 검증 그대로 + 새 묶음(kind · en_type)마다 10
    rng = random.Random(SEED); va_new = []
    groups = sorted({(r["kind"], r.get("en_type") or "") for r in new})
    for g in groups:
        rows = sorted([r for r in new if (r["kind"], r.get("en_type") or "") == g], key=lambda r: r["utt"]); rng.shuffle(rows); va_new += rows[:10]
    vid = {r["utt"] for r in va_new}
    tr = tr31 + [r for r in new if r["utt"] not in vid]; va = va31 + va_new
    for r in tr + va:
        r["lang"] = lang_of(r["text"])
    V4.mkdir(exist_ok=True); wj(V4 / "sref_train.jsonl", tr); wj(V4 / "sref_val.jsonl", va)
    allr = tr + va
    info = dict(name="SREF v4", created_kst=time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600)),
                base="SREF v3.1 − 사용자 '빼야 함' 4 + 새 묶음(인용 · 조사 조각 · 영어 숫자 · 날짜 · v3 탈락분 다시 시도) — 후보 풀에서 기존 필터 전부 통과한 것만",
                why="이사님 메일('~라고' 오류) · 사용자 청음 CSV(영어 English 태그 나음 18 · 빼야 함 4 · 숫자 영어 애매) — docs/plan/20261106_SREF_v4_notes.md",
                candidates=dict(voxcpm2="seed 1987 · 1988 · 1989(vox_s1~3)", sref2="SREF2_xvec_NSOSFT_01 s600 · seed 1986 · 1987 · 영어 = English 태그 · 코드스위치 = Korean 태그(통째)",
                                cap="SREF2 출신 ≤ 묶음의 절반"),
                user_drop=sorted(USER_DROP), user_drop_not_found=sorted(dropped), select=summ,
                counts=dict(train=len(tr), val=len(va), total=len(allr), new=len(new), by_kind=dict(Counter(r["kind"] for r in allr)),
                            by_src=dict(Counter(r.get("src_version", "?") for r in allr)), new_by_source=dict(Counter(r["gen_source"] for r in new)),
                            lang=dict(Counter(r["lang"] for r in allr)), val_by_kind=dict(Counter(r["kind"] for r in va)),
                            hours=round(sum(r["duration"] for r in allr) / 3600, 2)),
                v31_info="../v3_1/dataset_info.json", select_file="new/select.json")
    json.dump(info, open(V4 / "dataset_info.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(info["counts"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["sref2", "asr_en", "select", "build"])
    ap.add_argument("--cand", default=""); ap.add_argument("--seed", type=int, default=SEED); ap.add_argument("--shard", default="")
    a = ap.parse_args(); {"sref2": sref2, "asr_en": asr_en, "select": select, "build": build}[a.step](a)
