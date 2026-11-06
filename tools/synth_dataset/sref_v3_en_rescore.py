"""SREF v3 '영어만 조각'(en_only 800) 다시 거르기 → v3.1 (2026-11-06 · 사용자 결정 ①)
왜: 559개가 목소리 점수(WavLM < 0.60)로만 떨어졌다. 같은 길이에서도 영어는 한국어보다 0.15~0.2 낮다(채점기 언어 차이 가능성).
    사용자 청음: 목소리는 대체로 같지만 음질 · 억양이 어색한 것도 섞여 있음 → 목소리 문턱만 낮추고 다른 검사를 엄격하게.
새 규칙(en_only 800개 전부에 같은 규칙 · 기존 합격 114 도 다시 검사)
  · 목소리 sim ≥ 0.50 · 끝 잘림 없음
  · 영어 발음: whisper 언어 영어 고정 → CER ≤ 0.15(소문자 · 문장부호 제거 · 숫자는 영어 단어로 맞춤)
  · 억양 · 자연스러움(대리): whisper 평균 log 확률이 en_only 800 의 하위 20% 미만이면 제외
  · 음질: SQUIM ≥ v1 하위 5% 문턱(v2 와 같음) · 중간 무음 < 0.8초 · 클리핑 ≤ 1e-4 · DNSMOS 는 기록만(9초 창이라 2초 안팎 조각에 불리 — 1.5초 미만 3.0 이상 18% vs 2.5초 이상 67%)
단계
  1) CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/q3tts_ser/bin/python sref_v3_en_rescore.py asr
  2) ~/miniforge3/envs/q3tts/bin/python sref_v3_en_rescore.py filter      # DNSMOS(CPU) · 규칙 → en_rescore.json
  3) ~/miniforge3/envs/q3tts/bin/python sref_v3_en_rescore.py build       # v3_1/sref_{train,val}.jsonl · dataset_info.json
"""
import json, re, sys, time, random
from pathlib import Path
from collections import Counter
H = Path.home() / "workspaces"; S = H / "preprocessed/q3tts_koft/SREF"; N = S / "v3/new"; V3 = S / "v3"; V31 = S / "v3_1"
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
SIM_MIN, CER_MAX, LP_PCT, DNS_MIN, GAP_MAX = 0.50, 0.15, 20, None, 0.8   # DNS_MIN None = DNSMOS 는 기록만(짧은 소리에 불리 — 사용자 결정 11/06)


def en_ids():
    return [s["id"] for s in jl(N / "sentences.jsonl") if s["kind"] == "en_only"]


def asr():
    sys.path.insert(0, str(H / "q3tts_koft"))
    from ser.models import load_whisper, load_16k
    wh = load_whisper(); out = {}
    for i in en_ids():
        r = wh.transcribe(load_16k(str(N / "wav24" / f"{i}.wav")), language="en", temperature=0.0, condition_on_previous_text=False)
        segs = r.get("segments") or []
        lp = sum(s["avg_logprob"] * (s["end"] - s["start"]) for s in segs) / max(1e-6, sum(s["end"] - s["start"] for s in segs)) if segs else -9.0
        out[i] = dict(asr_en=r["text"].strip(), avg_logprob=round(lp, 4), no_speech=round(max([s["no_speech_prob"] for s in segs] or [0]), 3))
    json.dump(out, open(N / "asr_en.json", "w"), ensure_ascii=False, indent=1); print("ASR(en)", len(out))


def _norm_en(t):
    from ko_tts_norm.en_numbers import read_token
    t = t.lower().replace("’", "'")
    toks = []
    for w in re.findall(r"[a-z']+|\$?\d[\d:.,]*", t):
        if re.match(r"\$?\d", w):
            r = read_token(w.rstrip(".,")); toks.append((r or w).lower())
        else:
            toks.append(w)
    return re.sub(r"[^a-z]", "", " ".join(toks).replace("'", ""))


def _cer(r, h):
    d = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(h)] / max(1, len(r))


def filt():
    sys.path.insert(0, str(H / "q3tts_koft"))
    import numpy as np
    import subprocess
    S_ = {s["id"]: s for s in jl(N / "sentences.jsonl")}
    sc = json.load(open(N / "score.json")); Q = json.load(open(N / "quality.json")); A = json.load(open(N / "asr_en.json"))
    thr = json.load(open(S / "v1/quality_summary.json"))["thr"]
    ids = en_ids()
    tmp = N / "en_only_paths.jsonl"                          # DNSMOS 는 음질 측정 도구를 그대로 씀(CPU · 로컬 onnx)
    with open(tmp, "w") as f:
        for i in ids:
            f.write(json.dumps(dict(audio=str(N / "wav24" / f"{i}.wav"))) + "\n")
    subprocess.run([sys.executable, str(H / "q3tts_koft/tools/diag/audio_quality_survey.py"), "dnsmos", "--name", "sref_v3_en_only",
                    "--jsonl", str(tmp), "--workers", "32"], check=True, env={**__import__("os").environ, "OMP_NUM_THREADS": "1"})
    D = {r["path"]: r for r in jl(H / "outputs/q3tts_koft/eval/audio_quality/sref_v3_en_only.dnsmos.jsonl")}
    lp_thr = float(np.percentile([A[i]["avg_logprob"] for i in ids], LP_PCT))
    res = {}
    for i in ids:
        s, c, q, a = S_[i], sc[i], Q[i], A[i]; d = D[str(N / "wav24" / f"{i}.wav")]
        cer = _cer(_norm_en(s["text"]), _norm_en(a["asr_en"]))
        why = []
        if c["sim"] < SIM_MIN: why.append(f"목소리 {c['sim']:.2f}")
        if c["cut"]: why.append("끝 잘림")
        if cer > CER_MAX: why.append(f"영어 CER {cer:.2f}")
        if a["avg_logprob"] < lp_thr: why.append(f"ASR 확신도 {a['avg_logprob']:.2f}")
        bad_q = [k for k in ("stoi", "pesq", "sisdr") if q[k] < thr[k]] + (["긴 무음"] if q["max_gap"] >= GAP_MAX else []) + (["클리핑"] if q["clip"] > 1e-4 else [])
        why += bad_q
        if DNS_MIN is not None and d.get("dns_ovrl", 0) < DNS_MIN: why.append(f"DNSMOS {d.get('dns_ovrl', 0):.2f}")
        res[i] = dict(ok=not why, why=why, sim=c["sim"], cer_en=round(cer, 3), avg_logprob=a["avg_logprob"], dns_ovrl=d.get("dns_ovrl"),
                      squim={k: round(q[k], 3) for k in ("stoi", "pesq", "sisdr")}, asr_en=a["asr_en"], text=s["text"], dur=c["dur"])
    old = {r["utt"] for r in jl(V3 / "sref_train.jsonl") + jl(V3 / "sref_val.jsonl") if r["kind"] == "en_only"}
    summ = dict(rule=dict(sim_min=SIM_MIN, cer_en_max=CER_MAX, logprob_min=round(lp_thr, 4), logprob_pct=LP_PCT, dns_min=DNS_MIN, squim=thr),
                n=len(res), ok=sum(r["ok"] for r in res.values()), old_v3=len(old), old_still_ok=sum(res[i]["ok"] for i in old),
                reasons=dict(Counter(w.split()[0] for r in res.values() for w in r["why"])))
    json.dump(dict(summary=summ, items=res), open(N / "en_rescore.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


def build():
    R = json.load(open(N / "en_rescore.json")); items, summ = R["items"], R["summary"]
    tr3, va3 = jl(V3 / "sref_train.jsonl"), jl(V3 / "sref_val.jsonl")
    new_rows = {r["utt"]: r for r in jl(N / "train.jsonl") + [dict(x, why=None) for x in jl(N / "rejected.jsonl")]}
    Q = json.load(open(N / "quality.json"))
    keep = [i for i, r in items.items() if r["ok"]]
    en_rows = []
    for i in keep:
        r = dict(new_rows[i]); r.pop("why", None)
        r["squim"] = {k: round(Q[i][k], 3) for k in ("pesq", "stoi", "sisdr")}; r["max_gap"] = Q[i]["max_gap"]; r["src_version"] = "v3new"
        r["en_rescore"] = {k: items[i][k] for k in ("cer_en", "avg_logprob", "dns_ovrl", "asr_en")}
        en_rows.append(r)
    rng = random.Random(1986); ids = sorted(r["utt"] for r in en_rows); rng.shuffle(ids); val_ids = set(ids[:10])
    tr = [r for r in tr3 if r["kind"] != "en_only"] + [r for r in en_rows if r["utt"] not in val_ids]
    va = [r for r in va3 if r["kind"] != "en_only"] + [r for r in en_rows if r["utt"] in val_ids]
    V31.mkdir(exist_ok=True)
    for n, rr in (("sref_train.jsonl", tr), ("sref_val.jsonl", va)):
        with open(V31 / n, "w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    allr = tr + va
    info = dict(name="SREF v3.1", created_kst=time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600)),
                base="SREF v3 에서 '영어만 조각'(en_only)만 새 규칙으로 다시 거름 · 나머지는 v3 와 같음",
                why="사용자 결정 11/06 ①: 영어 조각은 목소리 점수가 언어 차이로 낮게 나옴 → 목소리 0.50 + 영어 고정 CER · ASR 확신도 · SQUIM · DNSMOS 로 엄격하게",
                en_rescore=summ, counts=dict(train=len(tr), val=len(va), total=len(allr), en_only=len(en_rows), by_kind=dict(Counter(r["kind"] for r in allr)),
                                             hours=round(sum(r["duration"] for r in allr) / 3600, 2)),
                v3_info="../v3/dataset_info.json", rescore_file="../v3/new/en_rescore.json")
    json.dump(info, open(V31 / "dataset_info.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(info["counts"], ensure_ascii=False))


if __name__ == "__main__":
    {"asr": asr, "filter": filt, "build": build}[sys.argv[1]]()
