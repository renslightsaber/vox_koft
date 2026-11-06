#!/usr/bin/env python
"""센 감정 칸 목소리 흔들림 줄이기 — ① 여러 번 만들어 고르기(best-of-N) + ③ 문턱 거르기 · ② 추론 노브 작게 시험 (2026-11-06).

왜: 청음에서 분노A · 기쁨B · 분노B(· 슬픔B) 출력이 "좀 다른 사람"처럼 들림. D안 데이터는 골라 쓰는 데이터라서,
    학습 없이 seed 를 여러 번 돌려 "그 화자다운 것"만 고르면 되는지 본다.
입력은 D 재측정(voxko/eval/d_eval.py)과 같음: "(v1 지시문)문장" + 목표 화자 무감정 reference · cfg 2.0 · timesteps 10 · 준비 합성 뒤.

세트
  bon  (①③) : 모델 LoRA01_s1500 · LoRA02iv2_s1500 × 목표 16명 × 칸 4(분노A · 기쁨B · 분노B · 슬픔B) × 문장 2 × seed 1986~1990
               seed 1986 = d_eval 합성음과 같은 소리인지 전부 비교(재현)
  knob (②)  : LoRA01_s1500 × 목표 4명(여2 · 남2 · aihub_0110 포함) × 같은 4칸 × 문장 2 × seed 1986
               조건 cfg15 · cfg20 · cfg25(reference 만) · prompt(reference + 같은 녹음을 prompt 로 · prompt_text = 그 녹음 전사 · cfg 2.0)
               · prompt_front(prompt 와 같되 지시문을 맨 앞에 — prompt 는 지시문을 소리 내어 읽어서)

단계(채점은 d_eval 함수를 그대로 불러 같은 잣대 · 결과는 ~/workspaces/tmp_listen/vox_koft/bestofn/)
  python tools/bestofn.py prep
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/voxcpm/bin/python    tools/bestofn.py vox --set bon --tag LoRA01_s1500
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/voxcpm/bin/python    tools/bestofn.py vox --set knob --tag cfg15      (cfg20 · cfg25 · prompt)
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/q3tts/bin/python     tools/bestofn.py sv  --set bon|knob   # 목표 1위 · REF 유사도 + 임베딩(WavLM · ERes2Net)
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/q3tts_ser/bin/python tools/bestofn.py ser --set bon|knob   # e2v · arousal · CER (+ 진짜 녹음 arousal 범위)
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/voxcpm/bin/python    tools/bestofn.py quality --set bon|knob
  ~/miniforge3/envs/voxcpm/bin/python tools/bestofn.py report     # 고르기 · 문턱 · 표 → bestofn/report.json · tables.md
  ~/miniforge3/envs/voxcpm/bin/python tools/bestofn.py page       # → bestofn/index.html
숫자는 모두 참고용 — 판정은 청음.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VOX_D_EVAL_OUT", str(Path.home() / "workspaces/tmp_listen/vox_koft/d_eval"))
from voxko.eval import d_eval as DE  # noqa: E402

H = Path.home() / "workspaces"
D = H / "tmp_listen/vox_koft/d_eval"
B = H / "tmp_listen/vox_koft/bestofn"
CK = H / "outputs/vox_koft/ckpt"
MODELS = {"LoRA01_s1500": CK / "A7N_ref_LoRA_01_r32/step_1500/export_lora",
          "LoRA02iv2_s1500": CK / "A7N_ref_LoRA_02_r32_iv2/step_1500/export_lora"}
CELLS = ["Angry-A", "Happy-B", "Angry-B", "Sad-B"]
SEEDS = [1986, 1987, 1988, 1989, 1990]
KNOB_MODEL = "LoRA01_s1500"
KNOB = {"cfg15": dict(cfg=1.5, prompt=False), "cfg20": dict(cfg=2.0, prompt=False),
        "cfg25": dict(cfg=2.5, prompt=False), "prompt": dict(cfg=2.0, prompt=True),
        # prompt 모드는 내부에서 text = prompt_text + 문장 → "(지시문)" 이 가운데 끼어 영어 지시문을 소리 내어 읽음(2026-11-06 확인)
        # → 지시문을 맨 앞(prompt_text 앞)에 둔 변형
        "prompt_front": dict(cfg=2.0, prompt=True, front=True)}
KNOB_SPK = ["a542_2214_G2A2E7_JCE", "aihub_0028", "a542_1837_G1A2E7_KHG", "aihub_0110"]   # 여 542 · 여 unseen8 · 남 542 · 남 unseen8(REF 먹먹)
SETS = {"bon": ("plan_bon.json", "wav_bon", "bon_"), "knob": ("plan_knob.json", "wav_knob", "knob_")}


def use_set(name: str):
    """d_eval 의 전역(출력 폴더 · 계획 · wav 폴더 · 앞붙이)을 이 세트로 돌린다 — d_eval 파일은 고치지 않음."""
    pf, wd, px = SETS[name]
    DE.OUT, DE.PLANF, DE.WAVD, DE.PFX = B, B / pf, B / wd, px
    return json.load(open(B / pf)) if (B / pf).exists() else None


# ───────────────────────────── prep
def prep(a):
    P = json.load(open(D / "plan.json"))
    base = [x for x in P["items"] if x["emo"] in CELLS]
    bon = [dict(x, id=f"{x['id']}_s{sd}", base_id=x["id"], seed=sd) for x in base for sd in SEEDS]
    knob = [dict(x, base_id=x["id"], seed=DE.SEED) for x in base if x["tgt"] in KNOB_SPK]
    B.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    for f, items, extra in (("plan_bon.json", bon, dict(models={k: str(v) for k, v in MODELS.items()}, seeds=SEEDS)),
                            ("plan_knob.json", knob, dict(model=KNOB_MODEL, conds=KNOB))):
        json.dump(dict(spk=P["spk"], distract=P["distract"], sents=P["sents"], items=items, cells=CELLS, created_utc=now, **extra),
                  open(B / f, "w"), ensure_ascii=False, indent=1)
    print(f"bon {len(bon)}/모델 · knob {len(knob)}/조건 · 목표 knob {KNOB_SPK}")


# ───────────────────────────── vox (env voxcpm)
def vox(a):
    import soundfile as sf
    from voxko.engine.synth import Engine
    P = use_set(a.set)
    front = False
    if a.set == "bon":
        model, cfg, use_prompt = MODELS[a.tag], 2.0, False
    else:
        model, cfg, use_prompt = MODELS[KNOB_MODEL], KNOB[a.tag]["cfg"], KNOB[a.tag]["prompt"]
        front = KNOB[a.tag].get("front", False)
    W = DE.WAVD / a.tag
    W.mkdir(parents=True, exist_ok=True)
    eng = Engine(str(model), emotion_map=str(DE.EMAP))                    # 준비 합성 포함
    t0, tot, n_new = time.time(), 0.0, 0

    def gen(x):
        s = P["spk"][x["tgt"]]
        kw = dict(prompt_wav_path=s["ref"], prompt_text=s["ref_text"]) if use_prompt else {}
        text = eng.compose(x["text"], emotion=x["emo"], normalize=False)
        if front:                                                         # "(지시문)prompt_text" + "문장"
            kw["prompt_text"] = text[: text.index(")") + 1] + s["ref_text"]
            text = x["text"]
        return eng.tts.generate(text=text, reference_wav_path=s["ref"],
                                cfg_value=cfg, inference_timesteps=10, max_len=a.max_len, normalize=False,
                                retry_badcase=False, seed=x["seed"], **kw)
    for x in P["items"]:
        p = W / f"{x['id']}.wav"
        if p.exists():
            continue
        w = gen(x)
        sf.write(p, w, eng.sr)
        tot += len(w) / eng.sr
        n_new += 1
        if n_new % 50 == 0:
            print(f"{a.tag} {n_new} · {time.time() - t0:.0f}s", flush=True)
    # 재현: d_eval 과 같은 조건(seed 1986 · cfg 2.0 · prompt 없음)이면 d_eval 합성음과 비교
    same = [x for x in P["items"] if x["seed"] == DE.SEED and cfg == 2.0 and not use_prompt]
    src = a.tag if a.set == "bon" else KNOB_MODEL
    rep = []
    for x in same:
        w1, w2 = sf.read(W / f"{x['id']}.wav")[0], sf.read(D / "wav" / src / f"{x['base_id']}.wav")[0]
        rep.append(len(w1) == len(w2) and float(np.abs(w1 - w2).max()) < 1e-3)
    meta = dict(tag=a.tag, set=a.set, model=str(model), cfg=cfg, prompt=use_prompt, front=front, n=len(P["items"]), new=n_new, max_len=a.max_len,
                repro_vs_d_eval=f"{sum(rep)}/{len(rep)}" if rep else None,
                rtf=(time.time() - t0) / tot if tot else None, time_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    json.dump(meta, open(B / f"{DE.PFX}vox_{a.tag}.json", "w"), indent=1)
    print(json.dumps(meta, ensure_ascii=False))


# ───────────────────────────── sv (env q3tts) — d_eval.sv + 임베딩
def sv(a):
    import librosa
    P = use_set(a.set)
    DE.sv(argparse.Namespace(tags=None, force=False))                  # sys.path 에 q3tts_koft 를 넣음
    from q3ko.eval.sv import load_embedder
    F = {"w": load_embedder("wavlm_large", device="cuda:0"), "e": load_embedder("eres2net", device="cuda:0")}
    for t in DE.tags():
        f = B / f"{DE.PFX}emb_{t}.npz"
        ids = [x["id"] for x in P["items"]]
        if f.exists() and list(np.load(f)["ids"]) == ids:
            continue
        W = [librosa.load(str(DE.WAVD / t / f"{i}.wav"), sr=16000)[0] for i in ids]
        np.savez(f, ids=np.array(ids), **{k: np.stack([F[k](w, 16000) for w in W]) for k in F})
        print("emb", t, len(ids), flush=True)


# ───────────────────────────── ser (env q3tts_ser) — d_eval.ser + 진짜 녹음 arousal 범위
def ser(a):
    use_set(a.set)
    DE.ser(argparse.Namespace(tags=None, force=False))
    f = B / "real_arousal.json"
    if f.exists():
        return
    from ser.models import dim_infer, e2v_infer, load_16k, load_dim, load_e2v
    e2v, dim = load_e2v(), load_dim()
    rows = [r for r in DE.rd(DE.VD / "val.jsonl") if r["emotion"] in CELLS]
    out = {}
    for r in rows:
        w = load_16k(r["audio"])
        labs, sc = e2v_infer(e2v, w)
        out[r["utt"]] = dict(emo=r["emotion"], top=labs[int(np.argmax(sc))], arousal=float(dim_infer(dim, w).get("arousal", np.nan)))
    json.dump(out, open(f, "w"), ensure_ascii=False, indent=1)
    print("진짜 녹음(A7N val) arousal", len(out))


# ───────────────────────────── quality (env voxcpm)
def quality(a):
    use_set(a.set)
    DE.quality(argparse.Namespace(tags=None, force=False, device="cuda"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prep", "vox", "sv", "ser", "quality", "report", "page"])
    ap.add_argument("--set", choices=list(SETS), default="bon")
    ap.add_argument("--tag")
    ap.add_argument("--max_len", type=int, default=600)
    a = ap.parse_args()
    if a.step in ("report", "page"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import bestofn_report as R  # 분석 · 페이지는 따로(길어서)
        return getattr(R, a.step)(a)
    {"prep": prep, "vox": vox, "sv": sv, "ser": ser, "quality": quality}[a.step](a)


if __name__ == "__main__":
    main()
