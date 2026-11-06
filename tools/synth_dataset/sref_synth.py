"""SREF(서비스 ref 단일 화자) 학습 데이터 합성 — VoxCPM2 무지시 · 서비스 ref(normalized 전사문) · seed 1986 (2026-11-05)

단계(단계마다 env 가 다름 · 데이터 폴더 OUT 하나를 공유):
  1) ~/miniforge3/envs/q3tts/bin/python    sref_synth.py pick  --out <OUT> --n_cs 100 --n_run117 50 --n_long 50 --n_a7n 50 --n_542 50
  2) CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/voxcpm/bin/python sref_synth.py synth --out <OUT> [--shard 0/4]
  3) CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/q3tts/bin/python sref_synth.py score --out <OUT>     # 목소리 유사도 · 끝 잘림 · 말 속도
  4) CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/q3tts_ser/bin/python sref_synth.py asr --out <OUT>   # whisper CER(한글 부분) · 영어 단어 확인
  5) ~/miniforge3/envs/q3tts/bin/python    sref_synth.py filter --out <OUT>                          # 합격 · 불합격 → train.jsonl
결과(OUT):
  sentences.jsonl(고른 문장 · 정규화) · wav48/<id>.wav(VoxCPM2 원본 48kHz) · wav24/<id>.wav(학습용 24kHz) · synth.json · score.json · asr.json
  train.jsonl(합격분: audio · text · ref_audio = 서비스 ref · speaker = service · kind) · rejected.jsonl(떨어진 이유) · filter_summary.json
학습 데이터로 쓰려면 다음으로 audio_codes 추출: (q3tts) python -m q3ko.data.extract_codes --jsonl <OUT>/train.jsonl
문장 원천: <SENT>/codeswitch.jsonl · customer_long.jsonl(Claude 작성) · run117 검수 고유 문장 · A7N · 542 학습 문장(raw_text 를 다시 정규화)
"""
import argparse, json, random, re, sys, time
from pathlib import Path

H = Path.home() / "workspaces"
SERVICE = H / "reference_audios/qwen3tts/set2_trim03_rms20/female_ment_natural_24k_trim03_rms20.wav"
SVC_TEXT = json.loads(SERVICE.with_suffix(".json").read_text(encoding="utf-8"))["normalized_text"]
SENT = H / "preprocessed/q3tts_koft/SREF/sentences"
A7 = H / "preprocessed/q3tts_koft/A7/preprocessed_outputs/train_cleanend.codes.jsonl"
S542 = H / "preprocessed/q3tts_koft/542/train_cleanend.codes.jsonl"
SEED = 1986
EXCLUDE = ["beta version"]                     # 사용자 요청: 「이 기능은 beta version 이라서 아직 불안정해요.」 류 제외
# 거르기 기준(시범 결과를 보고 조정)
CER_MAX, CER_MAX_CS, SIM_MIN, CPS = 0.10, 0.20, 0.60, (2.5, 9.0)   # CER_MAX_CS = 코드스위치(영어 철자 · 숫자 표기 차이로 whisper 오차 큼)   # 한글 CER · 서비스 ref 유사도(WavLM-large) · 초당 글자 수 범위
TRAIL_MIN, SIL_DB = 0.05, -45.0                  # 끝 잘림 = 끝 무음 < 0.05초 그리고 마지막 50ms > −45dB (q3ko.eval.metrics.tail_metrics 와 같음)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wj(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm_commit():
    import subprocess, ko_tts_norm
    d = Path(ko_tts_norm.__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "-C", str(d), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def pick(a):
    import ko_tts_norm
    rng = random.Random(SEED); out = []
    norm = lambda t: ko_tts_norm.normalize(t, profile=a.profile)
    bad = lambda t: any(x in t for x in EXCLUDE)
    # 코드스위치: en_type 별로 고르게
    cs = [r for f in a.cs_files.split(",") for r in jl(SENT / f) if not bad(r["text"])]
    by = {}
    for r in cs:
        by.setdefault(r.get("en_type", "x"), []).append(r)
    for v in by.values():
        rng.shuffle(v)
    k = 0
    while len([o for o in out if o["kind"] == "codeswitch"]) < min(a.n_cs, len(cs)):
        for t in sorted(by):
            if k < len(by[t]) and len([o for o in out if o["kind"] == "codeswitch"]) < a.n_cs:
                r = by[t][k]; out.append(dict(id=r["id"], kind="codeswitch", en_type=t, en=r.get("en", []), raw=r["text"]))
        k += 1
    # run117 고유 문장
    import pandas as pd
    R = next(H.glob("run117*")) / "results"
    s117 = set()
    for f in ("대화+음성_AB비교_전체.csv", "대화+음성검수_전체.csv"):
        s117 |= set(pd.read_csv(R / f)["챗봇 발화"].dropna().astype(str))
    s117 = sorted(t for t in s117 if not bad(t))
    for i, t in enumerate(rng.sample(s117, min(a.n_run117, len(s117)))):
        out.append(dict(id=f"r117_{i:04d}", kind="run117", raw=t))
    cl = [r for f in a.long_files.split(",") for r in jl(SENT / f)]
    for r in rng.sample(cl, min(a.n_long, len(cl))):
        out.append(dict(id=r["id"], kind="customer_long", raw=r["text"]))
    # 🆕 v3(11/06): 파일 전체를 kind 그대로(서빙 조각 모양 문장 · Claude 작성) — --all_files 가 비어 있으면 예전과 같음
    for f in [x for x in a.all_files.split(",") if x]:
        for r in jl(SENT / f):
            if not bad(r["text"]):
                out.append(dict(id=r["id"], kind=r["kind"], en_type=r.get("en_type"), en=r.get("en", []), raw=r["text"]))
    # 이미 쓴 A7N · 542 문장은 빼기(--exclude_used: 이전 판 sentences.jsonl 들 · v3 에서 사용)
    used = set()
    for p in [x for x in a.exclude_used.split(",") if x]:
        used |= {r["raw"] for r in jl(p)}
    # A7N · 542 학습 문장(무감정 읽기 · 2~10초 · 중복 없이)
    for path, n, kind in ((A7, a.n_a7n, "a7n"), (S542, a.n_542, "542")):
        seen, rows = set(used), []
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            t = r.get("raw_text") or r["text"]
            if 2.0 <= float(r["duration"]) <= 10.0 and t not in seen:
                seen.add(t); rows.append(t)
        for i, t in enumerate(rng.sample(sorted(rows), n)):
            out.append(dict(id=f"{kind}{a.id_tag}_{i:05d}", kind=kind, raw=t))
    nc = norm_commit()
    for o in out:
        o["text"] = norm(o["raw"]); o["profile"] = a.profile; o["normalizer_commit"] = nc
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    wj(od / "sentences.jsonl", out)
    from collections import Counter
    print("문장", len(out), dict(Counter(o["kind"] for o in out)))


def synth(a):
    import numpy as np, soundfile as sf, librosa
    from voxcpm import VoxCPM
    od = Path(a.out); w48, w24 = od / "wav48", od / "wav24"; w48.mkdir(exist_ok=True); w24.mkdir(exist_ok=True)
    S = jl(od / "sentences.jsonl")
    if a.shard:
        i, n = map(int, a.shard.split("/")); S = S[i::n]
    m = VoxCPM.from_pretrained(str(H / "ourckpts/voxcpm/VoxCPM2"), load_denoiser=False, optimize=False, local_files_only=True)
    sr = m.tts_model.sample_rate
    g = lambda t, seed: m.generate(text=t, prompt_wav_path=str(SERVICE), prompt_text=SVC_TEXT, reference_wav_path=str(SERVICE), seed=seed)
    g("준비 합성입니다.", SEED)                    # 첫 합성은 GPU 준비로 값이 조금 다름 → 버림
    meta, t0, tot = {}, time.time(), 0.0
    for s in S:
        p = w48 / f"{s['id']}.wav"
        if p.exists():
            continue
        seed = SEED + a.seed_offset
        w = g(s["text"], seed); sf.write(p, w, sr)
        sf.write(w24 / f"{s['id']}.wav", librosa.resample(np.asarray(w, np.float32), orig_sr=sr, target_sr=24000), 24000)
        meta[s["id"]] = dict(dur=len(w) / sr, seed=seed); tot += len(w) / sr
    # 재현 확인(첫 문장 다시)
    if S:
        w2 = g(S[0]["text"], SEED + a.seed_offset); w1 = sf.read(w48 / f"{S[0]['id']}.wav")[0]
        rep = len(w1) == len(w2) and float(np.abs(w1 - w2).max()) < 1e-3
    tag = a.shard.replace("/", "of") if a.shard else "all"
    json.dump(dict(meta=meta, repro=rep, rtf=(time.time() - t0) / max(tot, 1e-6), sr=sr), open(od / f"synth_{tag}.json", "w"), indent=1)
    print("합성", len(meta), "재현", rep, "RTF", round((time.time() - t0) / max(tot, 1e-6), 3))


def score(a):
    import numpy as np, librosa, soundfile as sf
    sys.path.insert(0, str(H / "q3tts_koft"))
    from q3ko.eval.sv import load_embedder
    from q3ko.eval.metrics import tail_metrics
    od = Path(a.out); S = jl(od / "sentences.jsonl")
    f = load_embedder("wavlm_large", device="cuda:0")
    ref = f(librosa.load(str(SERVICE), sr=16000)[0], 16000); res = {}
    for s in S:
        p = od / "wav24" / f"{s['id']}.wav"
        if not p.exists():
            continue
        y, sr = sf.read(p); trail, last_db = tail_metrics(str(p))
        nchar = len(re.sub(r"[^가-힣a-zA-Z0-9]", "", s["text"]))
        res[s["id"]] = dict(sim=float(ref @ f(librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000), 16000)),
                            dur=len(y) / sr, cps=nchar / max(len(y) / sr, 1e-3), trail=trail, last_db=last_db,
                            cut=bool(trail < TRAIL_MIN and last_db > SIL_DB))
    json.dump(res, open(od / "score.json", "w"), indent=1); print("채점", len(res))


def asr(a):
    sys.path.insert(0, str(H / "q3tts_koft"))
    from ser.models import load_16k, load_whisper
    od = Path(a.out); S = jl(od / "sentences.jsonl"); wh = load_whisper(); res = {}

    def _cer(r, h):
        d = list(range(len(h) + 1))
        for i, x in enumerate(r, 1):
            prev, d[0] = d[0], i
            for j, y in enumerate(h, 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
        return d[len(h)] / max(1, len(r))
    ko = lambda t: re.sub(r"[^가-힣0-9]", "", t)
    for s in S:
        p = od / "wav24" / f"{s['id']}.wav"
        if not p.exists():
            continue
        x = load_16k(str(p))
        en = re.findall(r"[A-Za-z][A-Za-z']+", s["text"])
        if en:   # 영어가 있으면 whisper 언어 자동(영어를 영어 철자로) · 한글+영어 소문자+숫자 전체 CER
            hyp = wh.transcribe(x, temperature=0.0)["text"]
            al = lambda t: re.sub(r"[^가-힣a-z0-9]", "", t.lower())
            r = dict(asr=hyp, cer_ko=_cer(al(s["text"]), al(hyp)), asr_lang="auto")
            hl = hyp.lower(); r["en_hit"] = sum(w.lower() in hl for w in en) / len(en)
        else:
            hyp = wh.transcribe(x, language="ko", temperature=0.0)["text"]
            r = dict(asr=hyp, cer_ko=_cer(ko(s["text"]), ko(hyp)), asr_lang="ko")
        res[s["id"]] = r
    json.dump(res, open(od / "asr.json", "w"), ensure_ascii=False, indent=1); print("ASR", len(res))


def filt(a):
    od = Path(a.out); S = jl(od / "sentences.jsonl")
    sc = json.load(open(od / "score.json")); ar = json.load(open(od / "asr.json"))
    syn, rtf, rep = {}, [], []
    for f in sorted(od.glob("synth_*.json")):
        d = json.load(open(f)); syn.update(d["meta"]); rtf.append(d["rtf"]); rep.append(d["repro"])
    ok, bad = [], []
    for s in S:
        i = s["id"]
        if i not in sc or i not in ar:
            continue
        c, r = sc[i], ar[i]; why = []
        cmax = CER_MAX_CS if (s["kind"] == "codeswitch" or s.get("en")) else CER_MAX   # v3: 영어가 든 묶음(en_only · en_q_embed …)도 코드스위치 문턱
        if r["cer_ko"] > cmax: why.append(f"CER {r['cer_ko']:.2f}")
        if c["sim"] < SIM_MIN: why.append(f"목소리 {c['sim']:.2f}")
        if c["cut"]: why.append("끝 잘림")
        if s.get("en_type") != "en_only" and s["kind"] != "en_only" and not CPS[0] <= c["cps"] <= CPS[1]: why.append(f"속도 {c['cps']:.1f}자/초")
        row = dict(audio=str(od / "wav24" / f"{i}.wav"), text=s["text"], ref_audio=str(SERVICE), ref_text=SVC_TEXT,
                   speaker="service", language="Auto", duration=round(c["dur"], 3), kind=s["kind"], utt=i, src_corpus="voxcpm2_sref",
                   raw_text=s["raw"], en_type=s.get("en_type"), profile=s.get("profile"), normalizer_commit=s.get("normalizer_commit"),
                   audio_48k=str(od / "wav48" / f"{i}.wav"), seed=syn.get(i, {}).get("seed", SEED),
                   score=dict(sim=round(c["sim"], 4), cer=round(r["cer_ko"], 4), cps=round(c["cps"], 2), trail=c["trail"], last_db=c["last_db"], asr=r["asr"]))
        (bad if why else ok).append(dict(row, why=why) if why else row)
    wj(od / "train.jsonl", ok); wj(od / "rejected.jsonl", bad)
    from collections import Counter
    kinds = sorted({s["kind"] for s in S})
    summ = dict(total=len(ok) + len(bad), passed=len(ok), rate=len(ok) / max(1, len(ok) + len(bad)),
                by_kind={k: [sum(r["kind"] == k for r in ok), sum(r["kind"] == k for r in ok + bad)] for k in kinds},
                reasons=dict(Counter(w.split()[0] for r in bad for w in r["why"])), hours=sum(r["duration"] for r in ok) / 3600,
                rule=dict(cer_ko_max=CER_MAX, cer_cs_max=CER_MAX_CS, sim_min=SIM_MIN, cps=CPS, endcut=f"trail<{TRAIL_MIN} & last50ms>{SIL_DB}dB"))
    json.dump(summ, open(od / "filter_summary.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(summ, ensure_ascii=False))
    info = dict(name=od.name, created_kst=time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600)),
                generator=dict(model="VoxCPM2", ckpt=str(H / "ourckpts/voxcpm/VoxCPM2"), mode="Ultimate cloning(prompt = reference = 서비스 ref, 지시문 없음)",
                               cfg_value=2.0, inference_timesteps=10, normalize=False, seed=SEED, warmup=True, repro=all(rep) if rep else None,
                               rtf=float(sum(rtf) / len(rtf)) if rtf else None, sr_out=48000, sr_train=24000, code=str(Path(__file__).resolve())),
                ref=dict(audio=str(SERVICE), text=SVC_TEXT, field="normalized_text"),
                text=dict(profiles=sorted({s.get("profile") for s in S}), normalizer_commits=sorted({s.get("normalizer_commit") for s in S}),
                          sources=dict(Counter(s["kind"] for s in S)), exclude=EXCLUDE),
                filter=summ, files=dict(train="train.jsonl", rejected="rejected.jsonl", sentences="sentences.jsonl", wav48="wav48/", wav24="wav24/"))
    json.dump(info, open(od / "dataset_info.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pick", "synth", "score", "asr", "filter"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_cs", type=int, default=100); ap.add_argument("--n_run117", type=int, default=50)
    ap.add_argument("--n_long", type=int, default=50); ap.add_argument("--n_a7n", type=int, default=50); ap.add_argument("--n_542", type=int, default=50)
    ap.add_argument("--profile", default="qwen3tts"); ap.add_argument("--cs_files", default="codeswitch.jsonl"); ap.add_argument("--long_files", default="customer_long.jsonl")
    ap.add_argument("--shard", default=""); ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--all_files", default="", help="v3: 이 파일들(SENT 기준, 쉼표)의 모든 행을 kind 그대로")
    ap.add_argument("--id_tag", default="", help="A7N · 542 id 에 붙일 표시(v3 = V3 → a7nV3_00000) — 이전 판 id 와 겹치지 않게")
    ap.add_argument("--exclude_used", default="", help="v3: 이 sentences.jsonl 들(절대경로, 쉼표)의 raw 는 A7N · 542 에서 다시 고르지 않음")
    a = ap.parse_args()
    {"pick": pick, "synth": synth, "score": score, "asr": asr, "filter": filt}[a.step](a)
