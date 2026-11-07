#!/usr/bin/env python
"""저장된 체크포인트로 검증 loss 를 다시 잰다 — 이전 run 의 val_xref(감정 ref 판) 비교용 (2026-11-07).

    CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/voxcpm/bin/python tools/val_xref_loss.py \
        --models LoRA01_s1500=A7N_ref_LoRA_01_r32:1500 A7NX_s3000=A7NX_xref_LoRA_01_r32:3000 base=base \
        [--jsonl ~/workspaces/preprocessed/vox_koft/A7NX/val_xref.jsonl] [--out ~/workspaces/tmp_listen/vox_koft/val_xref/val_xref_loss.json]

트레이너의 검증과 같은 계산: 같은 Dataset(make_dataset · 고정 ref · epoch 0) · EvalBatchSampler(seed 1986 · batch 8 · 1GPU 와 2GPU 가 같은 표본)
· 공식 packer + VoxCPM2Model.forward · bf16 autocast · preserved_rng(eval_seed) · loss = diff + stop 합 / 정답 칸 수 · CFM 잡음 seed 3개(1986~1988) 평균.
설정(지시문 표 · 비율 · max_seq_len · batch)은 그 run 의 ckpt/<run>/configs/merged.yaml 을 씀. base = 원본(LoRA_01 설정으로 데이터 구성).
검증: A7NX_xref_LoRA_01_r32 step 3000 을 다시 재서 TB val_xref/loss 와 같은지 본다(--check).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
H = Path.home() / "workspaces"
CK = H / "outputs/vox_koft/ckpt"
BASE = H / "ourckpts/voxcpm/VoxCPM2"


def merged(run: str) -> dict:
    t = "\n".join(l for l in open(CK / run / "configs/merged.yaml", encoding="utf-8") if not l.startswith("#"))
    return yaml.safe_load(t)


def load(run: str | None, step: int | None):
    from voxcpm.model.voxcpm2 import LoRAConfig, VoxCPM2Model
    if run is None:
        return VoxCPM2Model.from_local(str(BASE), optimize=False, training=True, device="cuda")
    d = CK / run / f"step_{step}"
    if (d / "export_lora").is_dir():
        cfg = json.loads((d / "export_lora/lora_config.json").read_text())
        m = VoxCPM2Model.from_local(cfg["base_model"], optimize=False, training=True, device="cuda", lora_config=LoRAConfig(**cfg["lora_config"]))
        loaded, skipped = m.load_lora_weights(str(d / "export_lora"))
        if skipped or not loaded:
            raise RuntimeError(f"LoRA 불러오기 이상: {len(loaded)} · 건너뜀 {len(skipped)}")
        return m
    return VoxCPM2Model.from_local(str(d / "export_ft"), optimize=False, training=True, device="cuda")


@torch.no_grad()
def measure(model, cfg: dict, jsonl: str) -> dict:
    from voxcpm.training.packers import AudioFeatureProcessingPacker
    from voxko.common.rng import preserved_rng
    from voxko.common.sampler import EvalBatchSampler
    from voxko.data.dataset import load_instruct_map, make_dataset
    dc, seed = cfg["data"], int(cfg["train"].get("seed", 1986)) if "seed" in cfg["train"] else int(cfg.get("seed", 1986))
    em = dc["emotion"]["map_file"]
    em = em if os.path.isabs(em) else str(ROOT / em)
    model.cuda()
    vae = model.audio_vae.cuda().eval()
    packer = AudioFeatureProcessingPacker(dataset_cnt=1, max_len=model.config.max_length, patch_size=model.config.patch_size,
                                          feat_dim=model.config.feat_dim, audio_vae=vae)
    ds = make_dataset(jsonl, None, model.text_tokenizer, load_instruct_map(em), dc, seed, "val_xref", strict=False)
    vs = EvalBatchSampler(ds.lengths, seed=seed, max_batches=None, batch_size=int(dc["batch_size"]), max_tokens=dc.get("max_tokens"),
                          num_replicas=1, rank=0, bucket_mult=int(dc["bucket_mult"]))
    lam = cfg["train"]["lambdas"]
    model.eval()
    s = dict(diff=0.0, stop=0.0, n=0.0)
    with preserved_rng(cfg["train"].get("eval_seed")):
        for idx in vs:
            b = ds.collate([ds[(i, 0)] for i in idx])
            b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            B = b["text_tokens"].size(0)
            with torch.autocast("cuda", enabled=False):
                p = packer(audio_tokens=b["audio_tokens"], text_tokens=b["text_tokens"], task_ids=torch.ones(B, dtype=torch.int32, device="cuda"),
                           dataset_ids=torch.zeros(B, dtype=torch.int32, device="cuda"), is_prompts=[False] * B,
                           ref_audio_tokens=b.get("ref_audio_tokens"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(p["text_tokens"], p["text_mask"], p["audio_feats"], p["audio_mask"], p["loss_mask"], p["position_ids"],
                            p["labels"], progress=1.0)
            n = float(p["loss_mask"].sum())
            s["diff"] += float(out["loss/diff"]) * n
            s["stop"] += float(out["loss/stop"]) * n
            s["n"] += n
    ld, ls = s["diff"] / s["n"], s["stop"] / s["n"]
    return dict(loss=float(lam["diff"]) * ld + float(lam["stop"]) * ls, loss_diff=ld, loss_stop=ls, frames=int(s["n"]), batches=vs.total, n=len(ds))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="이름=run:step | base=base")
    ap.add_argument("--jsonl", default=str(H / "preprocessed/vox_koft/A7NX/val_xref.jsonl"))
    ap.add_argument("--eval_seeds", type=int, nargs="+", default=[1986, 1987, 1988],
                    help="CFM 잡음 seed 들 — 평균을 씀(seed 하나로는 ±0.003 흔들림 · A7NX_s3000 실측 0.776 · 0.780 · 0.781 · TB 2GPU 0.779)")
    ap.add_argument("--out", default=str(H / "tmp_listen/vox_koft/val_xref/val_xref_loss.json"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.load(open(out)) if out.exists() else {}
    for spec in a.models:
        name, src = spec.split("=")
        if name in res:
            print(name, "이미 있음(건너뜀)", res[name])
            continue
        run, step = (None, None) if src == "base" else (src.split(":")[0], int(src.split(":")[1]))
        cfg = merged(run or "A7N_ref_LoRA_01_r32")
        t0 = time.time()
        model = load(run, step)
        per = {}
        for sd in a.eval_seeds:
            cfg["train"]["eval_seed"] = sd
            per[sd] = measure(model, cfg, a.jsonl)
        m = {k: float(sum(v[k] for v in per.values()) / len(per)) for k in ("loss", "loss_diff", "loss_stop")}
        m.update(per_seed={sd: round(v["loss"], 4) for sd, v in per.items()}, frames=per[a.eval_seeds[0]]["frames"],
                 batches=per[a.eval_seeds[0]]["batches"], n=per[a.eval_seeds[0]]["n"])
        m.update(run=run or "VoxCPM2 원본", step=step, sec=round(time.time() - t0, 1), jsonl=a.jsonl)
        res[name] = m
        json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
        print(name, json.dumps(m, ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
