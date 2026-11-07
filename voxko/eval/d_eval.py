"""D안 재측정 — "괄호 지시문으로 감정 + 처음 보는 목표 화자 목소리" 가 되는가 (vox_koft FT 의 성공 판정).

입력(D안 그대로): text = "(지시문)문장" · reference = 목표 화자 **무감정** 녹음 · prompt 없음(Controllable cloning).
목표 화자 16명 = 542 처음 보는 화자(val_unseen542) 여4 · 남4 + val_unseen8 8명. 감정 6칸 × 문장 2 = 화자당 12 → 모델당 192개.
문장 = A7N val(학습에 안 쓴 문장)에서 칸마다 2개 — 모든 목표 화자에 같은 12문장.

단계(각 단계는 그 env 의 python 으로 이 파일을 직접 실행):
  1) python voxko/eval/d_eval.py prep                                                           # plan.json
  2) CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/voxcpm/bin/python voxko/eval/d_eval.py vox --tag base --model ~/workspaces/ourckpts/voxcpm/VoxCPM2
     (FT/LoRA: --tag FT01_s3000 --model <ckpt>/step_3000/export_ft  |  --model <ckpt>/step_3000/export_lora)
  3) CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/q3tts/bin/python     voxko/eval/d_eval.py sv     # WavLM-large SV (q3tts_koft q3ko.eval.sv — 읽기만)
  4) CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/q3tts_ser/bin/python voxko/eval/d_eval.py ser    # emotion2vec · arousal · whisper CER (q3tts_koft ser/ — 읽기만)
  5) python voxko/eval/d_eval.py page                                                           # 청음 페이지 + summary.json
→ http://127.0.0.1:8765/vox_koft/d_eval/index.html   (산출물 ~/workspaces/tmp_listen/vox_koft/d_eval/)

목소리 잣대(WavLM-large · 코사인)
  · 화자 대표 = 그 화자 **진짜 녹음**(reference 로 쓴 파일 제외) 평균. 542 = val 2문장 · val_unseen8 = val 문장 최대 6개.
  · 후보 = 목표 16명 + 나머지 542 처음 보는 화자 134명(val 2문장) — **같은 성별** 후보끼리 순위.
  · 목표 1위 비율 = 출력이 같은 성별 후보 중 목표 화자에 가장 가까운 비율 · 목표 유사도 · reference 와의 유사도.
  · (대조) 진짜 녹음 = reference 파일 자체를 같은 방식으로 잰 값(완벽하면 나올 값).
감정 잣대: emotion2vec top 이 칸의 감정(happy · sad · angry)과 같은 비율 · arousal(audeering) · whisper CER(공백 · 기호 뺀 글자).
seed 1986 · cfg 2.0 · timesteps 10 · 준비 합성 1회 뒤 · normalize=False(A7N text 는 이미 ko_tts_norm).

2026-11-06 추가
  · prep --add_neutral : 중립(Neutrality) 칸 2문장을 기존 계획에 덧붙임(기존 192개는 그대로 · 화자당 14 → 모델당 224).
      A7N val 의 중립 문장은 텍스트가 모두 학습에 (다른 화자 녹음으로) 들어 있어서, 학습 텍스트에 **없는** A7N 계열 중립 문장
      (q3tts A val 2 + val_unseen8 8 = 10개, 녹음 2~6초)에서 고름.
  · prep --trial 20    : 가상 화자 시범 — 목표 16명 × 중립 20문장(위 10개 중 남은 것 + 542 val 문장, 모두 학습 텍스트에 없음) → plan_trial.json
  · --plan trial       : vox · sv · ser · quality 를 시범 계획으로(wav_trial/ · trial_*.json)
  · sv · ser · quality 는 **문항 단위로 이어서** 계산(새 문항만).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

H = Path.home() / "workspaces"
OUT = Path(os.environ.get("VOX_D_EVAL_OUT", H / "tmp_listen/vox_koft/d_eval"))
V542 = H / "preprocessed/q3tts_koft/542/val_unseen542"
VD = H / "preprocessed/vox_koft/A7N"
EMAP = H / "vox_koft/configs/emotion/a7n_vox_v1.yaml"
EMO6 = ["Happy-A", "Happy-B", "Sad-A", "Sad-B", "Angry-A", "Angry-B"]
EMO7 = ["Neutrality"] + EMO6                                                  # 2026-11-06 중립 칸 추가
KO = {"Neutrality": "중립", "Happy-A": "기쁨 A(약)", "Happy-B": "기쁨 B(강)", "Sad-A": "슬픔 A(약)", "Sad-B": "슬픔 B(강)", "Angry-A": "분노 A(약)", "Angry-B": "분노 B(강)"}
E2V = {"Neutrality": "neutral", "Happy": "happy", "Sad": "sad", "Angry": "angry"}
SEED = 1986
QA_VAL = H / "preprocessed/q3tts_koft/A/preprocessed_outputs/val.jsonl"
# --plan main(기본) | trial(가상 화자 중립 시범 생성) — 계획 파일 · wav 폴더 · 결과 파일 앞붙이
PLANF, WAVD, PFX = OUT / "plan.json", OUT / "wav", ""


def rd(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def base(e):
    return e.split("-")[0]


# ───────────────────────────── 1) prep
def prep(a):
    rng = random.Random(SEED)
    v542, r542 = rd(V542 / "val.jsonl"), rd(V542 / "refs.jsonl")
    val_by = defaultdict(list)
    for r in v542:
        val_by[r["speaker"]].append(r["audio"])
    ref_by = {r["speaker"]: r for r in r542}
    spk = {}
    for g in ("female", "male"):
        c = sorted(s for s, r in ref_by.items() if r["gender"] == g and len(val_by[s]) >= 2 and float(r["duration"]) <= 11.0)
        rng.shuffle(c)
        for s in c[:4]:
            spk[s] = dict(src="542", gender=g, ref=ref_by[s]["audio"], ref_text=ref_by[s]["text"], real=sorted(val_by[s]))
    uv, up = rd(VD / "val_unseen8/val.jsonl"), rd(VD / "val_unseen8/ref_pool.jsonl")
    ug = {r["speaker"]: r.get("gender") for r in uv}
    ureal = defaultdict(list)
    for r in uv:
        ureal[r["speaker"]].append(r["audio"])
    for p in sorted(up, key=lambda r: r["speaker"]):
        s = p["speaker"]
        spk[s] = dict(src="unseen8", gender=ug.get(s), ref=p["audio"], ref_text=p["text"],
                      real=[x for x in sorted(ureal[s]) if Path(x).stem != Path(p["audio"]).stem][:6])
    distract = {s: dict(gender=ref_by[s]["gender"], real=sorted(val_by[s])) for s in ref_by if s not in spk and len(val_by[s]) >= 2}
    val = rd(VD / "val.jsonl")
    sents = []
    for e in EMO6:
        c = sorted([r for r in val if r["emotion"] == e and 2.0 <= float(r["duration"]) <= 6.0], key=lambda r: r["utt"])
        for r in rng.sample(c, 2):
            sents.append(dict(emo=e, text=r["text"], src_utt=r["utt"]))
    items = [dict(id=f"{s}_{x['emo']}_{k % 2}", tgt=s, emo=x["emo"], text=x["text"]) for s in spk for k, x in enumerate(sents)]
    plan = dict(spk=spk, distract=distract, sents=sents, items=items, seed=SEED, created_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(plan, open(OUT / "plan.json", "w"), ensure_ascii=False, indent=1)
    print(f"목표 {len(spk)}명(542 {sum(v['src'] == '542' for v in spk.values())} · unseen8 {sum(v['src'] == 'unseen8' for v in spk.values())}) · "
          f"방해 후보 {len(distract)} · 문장 {len(sents)} · 합성 {len(items)}/모델")


def _unseen_neutral_pool():
    """학습 텍스트에 없는 A7N 계열 중립 문장(2~6초) — utt 순 정렬."""
    trt = {r["text"] for r in rd(VD / "train.jsonl")}
    c = [r for r in rd(QA_VAL) if r["emotion"] == "Neutrality"] + [r for r in rd(VD / "val_unseen8/val.jsonl") if r["emotion"] == "Neutrality"]
    c = [r for r in c if r["text"] not in trt]
    return sorted(c, key=lambda r: r["utt"]), trt


def prep_neutral(a):
    P = json.load(open(OUT / "plan.json"))
    if any(x["emo"] == "Neutrality" for x in P["sents"]):
        sys.exit("이미 중립 칸이 있습니다")
    pool, _ = _unseen_neutral_pool()
    c = [r for r in pool if 2.0 <= float(r["duration"]) <= 6.0]
    pick = random.Random(f"{SEED}|Neutrality").sample(c, 2)
    for k, r in enumerate(pick):
        P["sents"].append(dict(emo="Neutrality", text=r["text"], src_utt=r["utt"], text_src="A7N(학습 텍스트에 없음)"))
        P["items"] += [dict(id=f"{s}_Neutrality_{k}", tgt=s, emo="Neutrality", text=r["text"]) for s in P["spk"]]
    P["neutral_added_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    json.dump(P, open(OUT / "plan.json", "w"), ensure_ascii=False, indent=1)
    print("중립 2문장 추가:", [r["text"] for r in pick], "· 합성", len(P["items"]), "/모델")


def prep_trial(a):
    P = json.load(open(OUT / "plan.json"))
    used = {x["text"] for x in P["sents"]}
    pool, trt = _unseen_neutral_pool()
    sents = [dict(text=r["text"], src_utt=r["utt"], text_src="A7N") for r in pool if r["text"] not in used]
    v = sorted([r for r in rd(V542 / "val.jsonl") if 2.0 <= float(r["duration"]) <= 6.0 and r["text"] not in trt and r["text"] not in used],
               key=lambda r: r["utt"] if "utt" in r else r["audio"])
    need = a.trial - len(sents)
    sents += [dict(text=r["text"], src_utt=r.get("utt", Path(r["audio"]).stem), text_src="542") for r in random.Random(f"{SEED}|trial").sample(v, max(need, 0))]
    sents = sents[: a.trial]
    items = [dict(id=f"{s}_T{k:02d}", tgt=s, emo="Neutrality", text=x["text"]) for s in P["spk"] for k, x in enumerate(sents)]
    T = dict(spk=P["spk"], distract=P["distract"], sents=sents, items=items, seed=SEED, created_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    json.dump(T, open(OUT / "plan_trial.json", "w"), ensure_ascii=False, indent=1)
    print(f"시범 계획: 중립 {len(sents)}문장(A7N {sum(x['text_src'] == 'A7N' for x in sents)} · 542 {sum(x['text_src'] == '542' for x in sents)}) × 목표 {len(P['spk'])}명 = {len(items)}/모델")


# ───────────────────────────── 2) vox (env voxcpm)
def vox(a):
    import soundfile as sf
    sys.path.insert(0, str(H / "vox_koft"))
    from voxko.engine.synth import Engine
    P = json.load(open(PLANF))
    W = WAVD / a.tag
    W.mkdir(parents=True, exist_ok=True)
    eng = Engine(os.path.expanduser(a.model), emotion_map=str(EMAP))          # 준비 합성 포함
    ml = int(a.max_len)                                                       # 생성 최대 patch(0.16초) — 닿으면 "안 끝나는 합성"
    t0, tot, dur = time.time(), 0.0, {}
    chk = P["items"][5]
    for x in P["items"]:
        p = W / f"{x['id']}.wav"
        if p.exists():
            continue
        text = eng.compose(x["text"], emotion=x["emo"], normalize=False)
        w = eng.synth(text, ref=P["spk"][x["tgt"]]["ref"], seed=SEED, max_len=ml)
        sf.write(p, w, eng.sr)
        dur[x["id"]] = len(w) / eng.sr
        tot += len(w) / eng.sr
        chk = x                                                               # 재현 확인 = 이번에 새로 만든 마지막 문항
    x = chk
    w2 = eng.synth(eng.compose(x["text"], emotion=x["emo"], normalize=False), ref=P["spk"][x["tgt"]]["ref"], seed=SEED, max_len=ml)
    w1 = sf.read(W / f"{x['id']}.wav")[0]
    rep = len(w1) == len(w2) and float(np.abs(w1 - w2).max()) < 1e-3
    meta = dict(tag=a.tag, model=os.path.expanduser(a.model), kind=eng.kind, repro=bool(rep), repro_id=x["id"], n=len(P["items"]), max_len=ml,
                rtf=(time.time() - t0) / max(tot, 1e-9) if tot else None, time_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    json.dump(meta, open(OUT / f"{PFX}vox_{a.tag}.json", "w"), indent=1)
    print(json.dumps(meta, ensure_ascii=False))


def tags():
    return sorted(p.name for p in WAVD.iterdir() if p.is_dir())


def _todo(f, P, force, need_key=None):
    """이어서 계산: (기존 결과, 새로 계산할 문항)."""
    old = json.load(open(f)) if f.exists() and not force else {}
    if need_key and old and need_key not in next(iter(old.values())):
        old = {}
    return old, [x for x in P["items"] if x["id"] not in old]


# ───────────────────────────── 3) sv (env q3tts)
def sv(a):
    import librosa
    sys.path.insert(0, str(H / "q3tts_koft"))
    from q3ko.eval.sv import load_embedder
    P = json.load(open(PLANF))
    # 두 잣대(2026-11-06): WavLM-large(주) + ERes2Net — q3tts 의 "공정한 잣대"처럼 하나에 기대지 않도록 둘을 함께 본다
    F = {"w": load_embedder("wavlm_large", device="cuda:0"), "e": load_embedder("eres2net", device="cuda:0")}
    cache, wav16 = {}, {}

    def emb(p, k):
        p = str(p)
        if (k, p) not in cache:
            if p not in wav16:
                wav16[p] = librosa.load(p, sr=16000)[0]
            cache[(k, p)] = F[k](wav16[p], 16000)
        return cache[(k, p)]

    def mean(ps, k):
        v = np.mean([emb(p, k) for p in ps], 0)
        return v / np.linalg.norm(v)
    allspk = {**P["distract"], **P["spk"]}
    cand = {k: {s: (v["gender"], mean(v["real"], k)) for s, v in allspk.items()} for k in F}

    def judge(path, tgt):
        g, r = P["spk"][tgt]["gender"], {}
        for k, pre in (("w", ""), ("e", "e_")):
            o = emb(path, k)
            sims = {s: float(o @ c) for s, (gg, c) in cand[k].items() if gg == g}
            order = sorted(sims, key=lambda s: -sims[s])
            r.update({f"{pre}t_sim": sims[tgt], f"{pre}rank": order.index(tgt) + 1, f"{pre}top1": order[0] == tgt,
                      f"{pre}nearest": order[0], f"{pre}ref_sim": float(o @ emb(P["spk"][tgt]["ref"], k))})
        r["n_cand"] = len(order)
        return r
    real = {s: judge(v["ref"], s) for s, v in P["spk"].items()}              # (대조) reference 진짜 녹음
    json.dump(real, open(OUT / "sv_real.json", "w"), indent=1)
    for t in (a.tags or tags()):
        f_ = OUT / f"{PFX}sv_{t}.json"
        res, todo = _todo(f_, P, a.force, "e_top1")
        if not todo:
            print(t, "이미 있음(건너뜀)")
            continue
        res.update({x["id"]: judge(WAVD / t / f"{x['id']}.wav", x["tgt"]) for x in todo})
        json.dump(res, open(f_, "w"), indent=1)
        print(t, f"목표 1위 WavLM {100 * np.mean([r['top1'] for r in res.values()]):.0f}% · ERes2Net {100 * np.mean([r['e_top1'] for r in res.values()]):.0f}% · "
                 f"목표 유사도 {np.mean([r['t_sim'] for r in res.values()]):.3f}")
    print(f"(대조) 진짜 녹음 1위 WavLM {100 * np.mean([r['top1'] for r in real.values()]):.0f}% · ERes2Net {100 * np.mean([r['e_top1'] for r in real.values()]):.0f}%")


# ───────────────────────────── 4) ser (env q3tts_ser)
def ser(a):
    sys.path.insert(0, str(H / "q3tts_koft"))
    from ser.models import dim_infer, e2v_infer, load_16k, load_dim, load_e2v, load_whisper
    P = json.load(open(PLANF))
    e2v, dim, wh = load_e2v(), load_dim(), load_whisper()
    clean = lambda t: re.sub(r"[^가-힣a-zA-Z0-9]", "", t)  # noqa: E731

    def cer(r, h):
        d = list(range(len(h) + 1))
        for i, x in enumerate(r, 1):
            prev, d[0] = d[0], i
            for j, y in enumerate(h, 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
        return d[len(h)] / max(1, len(r))
    for t in (a.tags or tags()):
        f_ = OUT / f"{PFX}ser_{t}.json"
        res, todo = _todo(f_, P, a.force)
        if not todo:
            print(t, "이미 있음(건너뜀)")
            continue
        for x in todo:
            p = str(WAVD / t / f"{x['id']}.wav")
            w = load_16k(p)
            labs, sc = e2v_infer(e2v, w)
            d = dict(zip(labs, sc))
            dm = dim_infer(dim, w)
            hyp = wh.transcribe(w, language="ko", temperature=0.0, condition_on_previous_text=False)["text"]
            res[x["id"]] = dict(top=labs[int(np.argmax(sc))], happy=d.get("happy", 0), sad=d.get("sad", 0), angry=d.get("angry", 0),
                                neutral=d.get("neutral", 0), arousal=float(dm.get("arousal", np.nan)), valence=float(dm.get("valence", np.nan)),
                                cer=cer(clean(x["text"]), clean(hyp)), asr=hyp.strip())
        json.dump(res, open(f_, "w"), ensure_ascii=False, indent=1)
        print(t, f"감정 맞힘 {100 * np.mean([res[x['id']]['top'] == E2V[base(x['emo'])] for x in P['items']]):.0f}% · "
                 f"CER {100 * np.mean([r['cer'] for r in res.values()]):.1f}%")


# ───────────────────────────── 4-1) quality (env voxcpm · GPU) — 깨짐 · 이상
GAP_SEC, CLIP_FRAC, EMPTY_SEC = 0.8, 1e-4, 0.3


def quality(a):
    import soundfile as sf
    import torch
    import torchaudio
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    sys.path.insert(0, str(H / "vox_koft"))
    from voxko.eval.metrics import tail_metrics
    P = json.load(open(PLANF))
    dev = a.device
    m = SQUIM_OBJECTIVE.get_model().to(dev).eval()                             # 캐시 ~/.cache/torch/hub/torchaudio/models (받기 없음)
    for t in (a.tags or tags()):
        f_ = OUT / f"{PFX}quality_{t}.json"
        res, todo = _todo(f_, P, a.force)
        if not todo:
            print(t, "이미 있음(건너뜀)")
            continue
        vm = json.load(open(OUT / f"{PFX}vox_{t}.json")) if (OUT / f"{PFX}vox_{t}.json").exists() else {}
        cap = float(vm.get("max_len") or 4096) * 0.16                         # 생성 상한(초)
        for x in todo:
            y, sr = sf.read(WAVD / t / f"{x['id']}.wav", dtype="float32")
            dur = len(y) / sr
            with torch.no_grad():
                z = torchaudio.functional.resample(torch.from_numpy(y), sr, 16000)[None].to(dev)
                stoi, pesq, sisdr = (float(v) for v in m(z)) if z.shape[-1] > 1600 else (float("nan"),) * 3
            hop = sr // 100
            n = len(y) // hop
            peak = float(np.abs(y).max()) + 1e-12
            fr = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-12 if n else np.zeros(1)
            idx = np.where(20 * np.log10(fr / peak) > -40)[0]
            gap = float(np.diff(idx).max() - 1) * 0.01 if len(idx) > 1 else 0.0  # 말소리 사이 가장 긴 무음(초)
            voiced = (idx[-1] - idx[0] + 1) * 0.01 if len(idx) else 0.0
            syl = len(re.findall(r"[가-힣]", x["text"]))
            tm = tail_metrics(y, sr)
            res[x["id"]] = dict(dur=round(dur, 3), stoi=stoi, pesq=pesq, sisdr=sisdr, max_gap=round(max(gap, 0.0), 2),
                                clip=float((np.abs(y) >= 0.999).mean()), cut=bool(tm["cut_suspect"]), tail_sil_ms=tm["tail_sil_ms"],
                                empty=bool(dur < EMPTY_SEC or peak < 1e-3), hit_max=bool(dur >= cap - 0.5),
                                sps=round(syl / voiced, 2) if voiced > 0 else 0.0, syl=syl)
        json.dump(res, open(f_, "w"), indent=1)
        r = list(res.values())
        print(t, f"STOI {np.nanmean([v['stoi'] for v in r]):.3f} · PESQ {np.nanmean([v['pesq'] for v in r]):.2f} · "
                 f"긴 무음 {sum(v['max_gap'] >= GAP_SEC for v in r)} · 끝 잘림 {sum(v['cut'] for v in r)} · 빈 {sum(v['empty'] for v in r)} · "
                 f"최대 길이 {sum(v['hit_max'] for v in r)} · 말 속도 {np.mean([v['sps'] for v in r]):.2f} 음절/초")


# ───────────────────────────── 5) page
def _sq_thr():
    """깨짐 문턱값 = 원본(base) SQUIM 하위 5% — 원본보다 나쁜 꼬리를 센다."""
    f = OUT / "quality_base.json"
    if not f.exists():
        return None
    q = list(json.load(open(f)).values())
    return {k: float(np.nanpercentile([v[k] for v in q], 5)) for k in ("stoi", "pesq", "sisdr")}


def odd(v, thr) -> list:
    """한 출력의 이상 사유 목록(참고용)."""
    w = []
    if v["empty"]:
        w.append("빈 합성")
    if v["hit_max"]:
        w.append("안 끝남")
    if v["max_gap"] >= GAP_SEC:
        w.append("긴 무음")
    if v["clip"] > CLIP_FRAC:
        w.append("클리핑")
    if v["cut"]:
        w.append("끝 잘림")
    if thr and any(np.isfinite(v[k]) and v[k] < thr[k] for k in thr):
        w.append("SQUIM 낮음")
    return w


def summarize(P, t, thr=None):
    s, q = json.load(open(OUT / f"sv_{t}.json")), json.load(open(OUT / f"ser_{t}.json"))
    qa = json.load(open(OUT / f"quality_{t}.json")) if (OUT / f"quality_{t}.json").exists() else {}
    out = {}
    pct = lambda xs: 100 * float(np.mean(xs)) if len(xs) else float("nan")  # noqa: E731
    for grp, f in (("전체", lambda x: True), ("542 8명", lambda x: P["spk"][x["tgt"]]["src"] == "542"),
                   ("unseen8 8명", lambda x: P["spk"][x["tgt"]]["src"] == "unseen8")):
        xs = [x for x in P["items"] if f(x)]
        i = [x["id"] for x in xs]
        m = dict(n=len(xs), top1=pct([s[k]["top1"] for k in i]), t_sim=float(np.mean([s[k]["t_sim"] for k in i])),
                 ref_sim=float(np.mean([s[k]["ref_sim"] for k in i])), rank=float(np.median([s[k]["rank"] for k in i])),
                 emo_hit=pct([q[x["id"]]["top"] == E2V[base(x["emo"])] for x in xs]),
                 emo_hit6=pct([q[x["id"]]["top"] == E2V[base(x["emo"])] for x in xs if x["emo"] != "Neutrality"]),   # 6칸(어제 판과 비교용)
                 cer=pct([q[k]["cer"] for k in i]))
        if "e_top1" in s[i[0]]:
            m.update(e_top1=pct([s[k]["e_top1"] for k in i]), e_t_sim=float(np.mean([s[k]["e_t_sim"] for k in i])))
            m["fair_top1"] = (m["top1"] + m["e_top1"]) / 2                         # 공정한 잣대 = 두 잣대 평균
        if qa:
            m.update(odd=pct([bool(odd(qa[k], thr)) for k in i]), stoi=float(np.nanmean([qa[k]["stoi"] for k in i])),
                     pesq=float(np.nanmean([qa[k]["pesq"] for k in i])), sisdr=float(np.nanmean([qa[k]["sisdr"] for k in i])),
                     gap=pct([qa[k]["max_gap"] >= GAP_SEC for k in i]), cut=pct([qa[k]["cut"] for k in i]),
                     empty=pct([qa[k]["empty"] for k in i]), hit_max=pct([qa[k]["hit_max"] for k in i]),
                     clip=pct([qa[k]["clip"] > CLIP_FRAC for k in i]), sps=float(np.mean([qa[k]["sps"] for k in i])),
                     dur=float(np.mean([qa[k]["dur"] for k in i])))
        # 참고 점수(청음 전 후보 좁히기용): 감정 맞힘 + 목소리(공정) − CER − 이상 비율
        m["score"] = m["emo_hit"] + m.get("fair_top1", m["top1"]) - m["cer"] - m.get("odd", 0.0)
        i6 = [x["id"] for x in xs if x["emo"] != "Neutrality"]                 # 어제 판(6칸) 점수 — 순위 변화 확인용
        m["score6"] = (m["emo_hit6"] + (pct([s[k]["top1"] for k in i6]) + pct([s[k].get("e_top1", s[k]["top1"]) for k in i6])) / 2
                       - pct([q[k]["cer"] for k in i6]) - (pct([bool(odd(qa[k], thr)) for k in i6]) if qa else 0.0))
        out[grp] = m
    emos = [e for e in EMO7 if any(x["emo"] == e for x in P["items"])]
    out["칸별"] = {e: dict(emo_hit=pct([q[x["id"]]["top"] == E2V[base(e)] for x in P["items"] if x["emo"] == e]),
                          arousal=float(np.mean([q[x["id"]]["arousal"] for x in P["items"] if x["emo"] == e])),
                          top1=pct([s[x["id"]]["top1"] for x in P["items"] if x["emo"] == e]),
                          t_sim=float(np.mean([s[x["id"]]["t_sim"] for x in P["items"] if x["emo"] == e])),
                          cer=pct([q[x["id"]]["cer"] for x in P["items"] if x["emo"] == e]),
                          **({"e_top1": pct([s[x["id"]]["e_top1"] for x in P["items"] if x["emo"] == e])} if "e_top1" in s[P["items"][0]["id"]] else {}),
                          **({"odd": pct([bool(odd(qa[x["id"]], thr)) for x in P["items"] if x["emo"] == e])} if qa else {}))
                  for e in emos}
    return out


def ready_tags():
    ids = {x["id"] for x in json.load(open(OUT / "plan.json"))["items"]}
    ok = lambda f: f.exists() and ids <= set(json.load(open(f)))  # noqa: E731
    T = [t for t in tags() if ok(OUT / f"sv_{t}.json") and ok(OUT / f"ser_{t}.json")]
    miss = [t for t in tags() if t not in T]
    if miss:
        print("⚠️ 문항이 덜 채점된 이름표(페이지에서 뺌):", miss)
    return sorted(T, key=lambda t: (t != "base", t))                           # 원본을 맨 앞에


def page(a):
    import soundfile as sf
    import yaml
    P = json.load(open(OUT / "plan.json"))
    T_all = ready_tags()
    thr = _sq_thr()
    S = {t: summarize(P, t, thr) for t in T_all}
    T = [t for t in T_all if not a.page_tags or t in a.page_tags]
    SV = {t: json.load(open(OUT / f"sv_{t}.json")) for t in T}
    SR = {t: json.load(open(OUT / f"ser_{t}.json")) for t in T}
    QA = {t: (json.load(open(OUT / f"quality_{t}.json")) if (OUT / f"quality_{t}.json").exists() else {}) for t in T}
    real = json.load(open(OUT / "sv_real.json"))
    R = OUT / "ref"
    R.mkdir(exist_ok=True)

    def cp(src, s):
        d = R / f"{s}_ref.wav"
        if not d.exists():
            import librosa
            y, _ = librosa.load(src, sr=24000)
            sf.write(d, y, 24000)
        return f"ref/{s}_ref.wav"
    E = html.escape
    A = lambda p: f'<audio controls preload="none" src="{p}"></audio>'  # noqa: E731

    def btn(i, opts):
        return "<div class=b data-id='%s'>%s</div>" % (E(i), "".join(f"<button data-v='{o}'>{o}</button>" for o in opts))
    instr = (yaml.safe_load(open(EMAP, encoding="utf-8")) or {}).get("instruct", {})
    first = lambda v: v[0] if isinstance(v, list) else v  # noqa: E731
    h = [f"""<!doctype html><meta charset=utf-8><title>vox_koft D안 재측정</title>
<style>body{{font-family:sans-serif;max-width:1800px;margin:20px auto;line-height:1.45}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:5px;vertical-align:top;font-size:.88em}}th{{background:#f3f3f3}}.n{{color:#666;font-size:.8em}}audio{{width:170px}}button{{margin:1px;font-size:.8em}}button.on{{background:#1565c0;color:#fff}}#bar{{position:sticky;top:0;background:#fff;padding:6px;border-bottom:1px solid #ccc;z-index:9}}.num{{display:none}}body.shownum .num{{display:block}}details{{margin:6px 0}}</style>
<div id=bar>판정 <span id=cnt>0</span>개 · <button onclick="dl()">CSV 내려받기</button> · <label><input type=checkbox onchange="document.body.classList.toggle('shownum',this.checked)"> 숫자 보기(먼저 귀로!)</label> · <button onclick="if(confirm('판정을 모두 지울까요?')){{localStorage.removeItem(K);location.reload()}}">판정 지우기</button></div>
<h1>🎧 vox_koft D안 재측정 — 지시문으로 감정 + 처음 보는 목표 화자 목소리</h1>
<h2>청음 가이드</h2><ul>
<li>입력은 모든 열이 같습니다: <b>"(영어 감정 지시문 — v1 문구)문장"</b> + 목표 화자의 <b>무감정</b> 녹음(reference) · prompt 없음 · seed {SEED}.</li>
<li>열: <b>VoxCPM2 원본(base)</b> → FT · LoRA 후보(이름표 = run 줄임 _ step). 출력마다 두 가지를 판정합니다.
 <ul><li><b>목소리</b>: 목표(reference 와 같은 사람) · 애매 · 다름</li><li><b>감정</b>: OK(칸의 감정이 분명) · 약함 · 틀림</li></ul></li>
<li><b>성공</b> = 원본보다 감정이 확실히 세지고, 목소리는 원본만큼 목표를 따라감. 칸당 1~2개라 <b>확실히 다른 칸이 3칸 이상</b>일 때만 차이로 봅니다.</li>
<li>A = 감정을 누르는 쪽(약), B = 드러내는 쪽(강). <b>중립</b> 칸(2026-11-06 추가)은 v1 의 Neutrality 지시문 + 학습 텍스트에 없는 문장입니다. 판정은 이 브라우저에 저장되고 CSV 로 받을 수 있습니다.</li>
<li>숫자(자동 잣대 · <b>참고용</b>)는 맨 아래 접힌 곳과 "숫자 보기"에 있습니다 — <b>귀로 먼저 고른 뒤</b> 보세요.</li>
<li>청음 열: {', '.join(E(t) for t in T)} · 요약 표에는 평가한 모든 step({len(T_all)}개)이 있습니다.</li></ul>"""]
    for f in ("page_top.html",):                                                # 다른 도구가 만든 덧붙임(맨 위 요약 — 접힘)
        if (OUT / f).exists():
            h.append((OUT / f).read_text(encoding="utf-8"))
    h.append("<h2>① 목표 화자별 — 감정 7칸 × 2문장</h2>")
    RC = json.load(open(OUT / "real_cells.json"))["cells"] if (OUT / "real_cells.json").exists() else {}   # tools/real_cells.py (2026-11-07)
    if RC:
        h.append("<p class=n>unseen8 화자 표에는 <b>진짜 사람 녹음</b> 열이 있습니다 — 그 사람이 실제로 그 칸 감정을 낸 녹음 1개(비교용 · 말투 · 문장 다름 · 없는 칸은 '없음'). "
                 "고른 방법: tools/real_cells.py.</p>")
    for s_, v in P["spk"].items():
        xs = sorted([x for x in P["items"] if x["tgt"] == s_], key=lambda x: (EMO7.index(x["emo"]), x["id"]))
        h.append(f"<details><summary><b>{E(s_)}</b> ({'여' if v['gender'] == 'female' else '남'} · {v['src']}) — {len(xs)}문장</summary>")
        h.append(f"<p>목표 reference {A(cp(v['ref'], s_))} <span class=n>“{E(v['ref_text'])}”</span></p>")
        rc = RC and v["src"] == "unseen8"
        h.append("<table><tr><th>칸</th><th>문장</th>" + ("<th>진짜 사람 녹음<br><span class=n>(비교용 · 말투 · 문장 다름)</span></th>" if rc else "")
                 + "".join(f"<th>{E(t)}</th>" for t in T) + "</tr>")
        seen = set()
        for x in xs:
            i = x["id"]
            cells = []
            for t in T:
                r, q, qa = SV[t][i], SR[t][i], QA[t].get(i)
                rk = "✓1위" if r["top1"] else f"{r['rank']}위"
                ek = (" · E " + ("✓1위" if r["e_top1"] else f"{r['e_rank']}위")) if "e_top1" in r else ""
                why = (" · ⚠️" + "/".join(odd(qa, thr))) if qa and odd(qa, thr) else ""
                num = (f"<div class='n num'>목표 {r['t_sim']:.2f} · {rk}{ek}<br>e2v {q['top']} · arousal {q['arousal']:.2f} · CER {100 * q['cer']:.0f}%"
                       + (f"<br>{qa['dur']:.1f}초 · {qa['sps']:.1f}음절/초{why}" if qa else "") + "</div>")
                cells.append(f"<td>{A(f'wav/{t}/{i}.wav')}{num}{btn(f'{t}|{i}|v', ['목표', '애매', '다름'])}{btn(f'{t}|{i}|e', ['OK', '약함', '틀림'])}</td>")
            rcell = ""
            if rc and x["emo"] not in seen:                                     # 칸마다 1개 — 그 칸 첫 줄에 rowspan
                seen.add(x["emo"])
                k = RC.get(f"{s_}|{x['emo']}")
                n = sum(y["emo"] == x["emo"] for y in xs)
                rcell = (f"<td rowspan={n}>" + (f"{A(k['file'])}<div class=n>“{E(k['text'])}”</div>" if k else "<span class=n>없음</span>") + "</td>")
            h.append(f"<tr><td>{KO[x['emo']]}</td><td>{E(x['text'])}<div class=n>({E(first(instr.get(x['emo'], '')))})</div></td>{rcell}{''.join(cells)}</tr>")
        h.append("</table></details>")
    if (OUT / "page_vset.html").exists():                                       # ② 가상 화자 세트(tools/vspk.py)
        h.append((OUT / "page_vset.html").read_text(encoding="utf-8"))
    h.append("<details><summary><b>📊 자동 잣대 요약 — 참고용 (귀로 판정한 뒤 열기)</b></summary>")
    rr = list(real.values())
    h.append(f"<p class=n>(대조) 목표 화자 진짜 녹음(reference)을 같은 잣대로: 목표 1위 WavLM {100 * np.mean([r['top1'] for r in rr]):.0f}%"
             + (f" · ERes2Net {100 * np.mean([r['e_top1'] for r in rr]):.0f}%" if 'e_top1' in rr[0] else "") + " — 완벽하면 나올 값</p>")
    cols = [("top1", "목표 1위<br>WavLM"), ("e_top1", "목표 1위<br>ERes2Net"), ("t_sim", "목표 유사도"), ("emo_hit", "e2v 감정"),
            ("cer", "CER"), ("odd", "이상 비율"), ("stoi", "STOI"), ("pesq", "PESQ"), ("sps", "음절/초"), ("score", "참고 점수")]
    h.append("<table><tr><th>모델</th><th>묶음</th>" + "".join(f"<th>{c}</th>" for _, c in cols) + "</tr>")
    fmt = lambda k, v: "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.3f}" if k in ("t_sim", "stoi") else f"{v:.2f}" if k in ("pesq", "sps") else f"{v:.0f}" + ("" if k == "score" else "%"))  # noqa: E731
    for t in T_all:
        for g in ("전체", "542 8명", "unseen8 8명"):
            m = S[t][g]
            h.append(f"<tr><td>{E(t)}</td><td>{g}</td>" + "".join(f"<td>{fmt(k, m.get(k))}</td>" for k, _ in cols) + "</tr>")
    emos = [e for e in EMO7 if any(x["emo"] == e for x in P["items"])]
    h.append("</table><table><tr><th>모델</th>" + "".join(f"<th>{KO[e]}<br>e2v · arousal · 목표1위</th>" for e in emos) + "</tr>")
    for t in T_all:
        h.append(f"<tr><td>{E(t)}</td>" + "".join(f"<td>{S[t]['칸별'][e]['emo_hit']:.0f}% · {S[t]['칸별'][e]['arousal']:.2f} · {S[t]['칸별'][e]['top1']:.0f}%</td>" for e in emos) + "</tr>")
    h.append(f"""</table><p class=n>목표 1위 = 같은 성별 후보(목표 16명 + 542 처음 보는 화자 134명) 중 출력이 목표 화자 진짜 녹음 평균에 가장 가까운 비율.
공정한 잣대 = WavLM-large · ERes2Net 두 잣대의 평균. 이상 = 빈 합성 · 안 끝남 · 긴 무음(≥{GAP_SEC}초) · 클리핑 · 끝 잘림 · SQUIM 이 원본 하위 5% 보다 낮음.
참고 점수 = e2v 감정 + 목표 1위(공정) − CER − 이상 비율(2026-11-06 부터 중립 칸 포함 7칸 · 중립 = e2v 가 neutral 을 고르면 맞힘). emotion2vec 은 한국어 슬픔을 잘 못 잡습니다 — 귀가 정본.</p></details>""")
    h.append("""<script>const K='vox_koft_d_eval';let S=JSON.parse(localStorage.getItem(K)||'{}');
function draw(){document.querySelectorAll('.b').forEach(d=>{d.querySelectorAll('button').forEach(b=>b.classList.toggle('on',S[d.dataset.id]===b.dataset.v))});document.getElementById('cnt').textContent=Object.keys(S).length}
document.querySelectorAll('.b button').forEach(b=>b.onclick=()=>{const id=b.parentElement.dataset.id;S[id]=S[id]===b.dataset.v?undefined:b.dataset.v;if(!S[id])delete S[id];localStorage.setItem(K,JSON.stringify(S));draw()});
function dl(){const r=['model,id,axis,value'].concat(Object.entries(S).map(([k,v])=>k.split('|').join(',')+','+v));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+r.join('\\n')],{type:'text/csv'}));a.download='vox_koft_d_eval_judgments.csv';a.click()}
draw()</script>""")
    (OUT / "index.html").write_text("\n".join(h), encoding="utf-8")
    json.dump(dict(models=S, squim_thr=thr, real=dict(top1=100 * np.mean([r['top1'] for r in rr]), t_sim=float(np.mean([r['t_sim'] for r in rr])))),
              open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("✅ page →", OUT / "index.html", "· 청음 열", T)
    for t in T_all:
        m = S[t]["전체"]
        print(f"  {t:<22} 목표1위 W {m['top1']:.0f}% E {m.get('e_top1', float('nan')):.0f}% · 감정 {m['emo_hit']:.0f}% · CER {m['cer']:.1f}% · "
              f"이상 {m.get('odd', float('nan')):.0f}% · 점수 {m['score']:.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prep", "vox", "sv", "ser", "quality", "page"])
    ap.add_argument("--tag", help="vox: 모델 이름표(예: base · FT01_s3000)")
    ap.add_argument("--model", help="vox: 원본 · export_ft · export_lora 폴더")
    ap.add_argument("--tags", nargs="*", help="sv · ser · quality: 이 이름표만(기본: wav/ 아래 전부)")
    ap.add_argument("--max_len", type=int, default=600, help="vox: 생성 최대 patch(0.16초 · 600 = 96초) — 닿으면 '안 끝나는 합성'")
    ap.add_argument("--force", action="store_true", help="sv · ser · quality: 이미 있어도 다시 계산")
    ap.add_argument("--device", default="cuda", help="quality: cuda | cpu")
    ap.add_argument("--page_tags", nargs="*", help="page: 청음 표에 넣을 이름표(기본: 전부 · 요약 표는 늘 전부)")
    ap.add_argument("--add_neutral", action="store_true", help="prep: 기존 계획에 중립 칸 2문장 덧붙이기")
    ap.add_argument("--trial", type=int, default=0, help="prep: 가상 화자 시범 계획(목표 16명 × 중립 N문장) → plan_trial.json")
    ap.add_argument("--plan", choices=["main", "trial"], default="main", help="vox · sv · ser · quality: 계획(trial = wav_trial/ · trial_*.json)")
    a = ap.parse_args()
    global PLANF, WAVD, PFX
    if a.plan == "trial":
        PLANF, WAVD, PFX = OUT / "plan_trial.json", OUT / "wav_trial", "trial_"
    if a.step == "prep" and (a.add_neutral or a.trial):
        return prep_neutral(a) if a.add_neutral else prep_trial(a)
    if a.step == "vox" and not (a.tag and a.model):
        sys.exit("vox 는 --tag 와 --model 이 필요합니다")
    {"prep": prep, "vox": vox, "sv": sv, "ser": ser, "quality": quality, "page": page}[a.step](a)


if __name__ == "__main__":
    main()
