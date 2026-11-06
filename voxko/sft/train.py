#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VoxCPM2 한국어 감정 전체 FT 학습 진입점.

권장 실행 — 공통 런처(hw ↔ accelerate 짝 · 로그 저장 · 실행 기록):
    ./scripts/run.sh train sft --exp configs/experiments/sft/<ID>.yaml --hw rtxpro6000_1gpu --gpu 0 --smoke
    ./scripts/run.sh train sft --exp configs/experiments/sft/<ID>.yaml --hw rtxpro6000_2gpu --accel 2gpu_a

직접 실행:
    accelerate launch --config_file configs/accelerate/rtxpro6000_4gpu.yaml \\
        -m voxko.sft.train --hardware rtxpro6000_4gpu --experiment configs/experiments/sft/<ID>.yaml

옵션
    --smoke                  TB · 저장 끔, smoke_steps step + val 2배치 + 합성 1개 + export 검증
    --resume_from_latest     마지막 재개용 step_* 에서 이어서(데이터 위치 · RNG 까지 복원)
    --allow_schedule_change  resume 때 max_steps · scheduler · warmup 이 체크포인트와 달라도 진행(곡선이 바뀜)
    --force_overwrite        기존 체크포인트가 있어도 처음부터(권장하지 않음)
    --set key=value          값 하나 덮어쓰기(여러 번) · --overlay <yaml> · --max_steps N
"""
from __future__ import annotations

from voxko.common import console  # noqa: F401  ★ 가장 먼저(colorama → logging → tqdm 순서)

import argparse
import sys

from voxko.common.config import ConfigError, load_config


def main(argv=None):
    ap = argparse.ArgumentParser(description="VoxCPM2 한국어 전체 FT (accelerate)")
    ap.add_argument("--hardware", "--hw", required=True, help="rtxpro6000_<N>gpu 또는 yaml 경로 (필수)")
    ap.add_argument("--experiment", "--exp", required=True, help="configs/experiments/sft/<ID>.yaml")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--overlay", action="append", default=[], help="추가 yaml(순서대로 덮어씀)")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume_from_latest", "--resume", action="store_true")
    ap.add_argument("--allow_schedule_change", action="store_true")
    ap.add_argument("--force_overwrite", action="store_true")
    args = ap.parse_args(argv)

    overrides = {"train": {"max_steps": args.max_steps}} if args.max_steps is not None else None
    try:
        cfg = load_config(args.hardware, args.experiment, args.config, args.overlay, args.sets, overrides)
    except ConfigError as e:
        console.say(f"❌ {e}", "RED", "BRIGHT")
        sys.exit(2)

    from voxko.sft.trainer import VoxSFTTrainer
    VoxSFTTrainer(cfg, smoke=args.smoke, resume=args.resume_from_latest, force_overwrite=args.force_overwrite,
         allow_schedule_change=args.allow_schedule_change).fit()


if __name__ == "__main__":
    main()
