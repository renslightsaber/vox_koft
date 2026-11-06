"""학습 전 설정 백업 — exp_reports/<sft|lora>/<run>/config_backup/ (q3tts_koft 와 같은 형식).

    python tools/pre_train_backup.py configs/experiments/sft/A7N_ref_FT_01.yaml --hw rtxpro6000_2gpu --accel 2gpu_a

만드는 것: 원본 yaml(config · hardware · experiment) · accelerate 파일 · 감정 지시문 표 · merged.yaml(병합 결과, 머리에 날짜)
· md5.txt. 학습이 시작되면 ckpt/<run>/configs/merged.yaml 이 자동으로 생기며 이것과 같아야 한다(학습 뒤 확인).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

import yaml

from voxko.common.config import load_config, public_view
from voxko.common.paths import REPO_ROOT


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp")
    ap.add_argument("--hw", required=True)
    ap.add_argument("--accel", default=None, help="2gpu_a | 2gpu_b (2GPU 일 때)")
    ap.add_argument("--force", action="store_true", help="이미 있으면 덮어쓰기")
    a = ap.parse_args(argv)
    cfg = load_config(a.hw, a.exp)
    run = cfg["_derived"]["run_name"]
    track = "lora" if cfg["lora"]["enable"] else "sft"
    out = REPO_ROOT / "exp_reports" / track / run / "config_backup"
    if out.exists() and not a.force:
        sys.exit(f"이미 있습니다: {out} (--force 로 덮어쓰기)")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    hw = a.hw
    acc = REPO_ROOT / "configs/accelerate" / (f"{hw[:-5]}_{a.accel}.yaml" if hw.endswith("_2gpu") else f"{hw}.yaml")
    files = [(f"{role}__{Path(src).name}", Path(src)) for role, src in cfg["_run"]["sources"].items()]
    files += [(f"accelerate__{acc.name}", acc), (f"emotion_map__{Path(cfg['_derived']['emotion_map']).name}",
                                                 Path(cfg["_derived"]["emotion_map"]))]
    lines = []
    for name, src in files:
        shutil.copy2(src, out / name)
        lines.append(f"{md5(src)}  {name}  ← {src}")
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with open(out / "merged.yaml", "w", encoding="utf-8") as f:
        f.write(f"# {run} 학습 전 백업({ts} UTC) — config → hardware({hw}) → experiment 병합 결과. 🔒 수정 금지\n"
                "# 실제 학습 때는 run 폴더 configs/merged.yaml 도 자동 저장됩니다(이것과 같아야 함).\n")
        yaml.safe_dump(public_view(cfg), f, allow_unicode=True, sort_keys=False)
    lines.append(f"{md5(out / 'merged.yaml')}  merged.yaml")
    (out / "md5.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    for ln in lines:
        print("  " + ln)


if __name__ == "__main__":
    main()
