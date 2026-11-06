"""VoxCPM 체크포인트 다운로드 (vox_koft) — HF → ~/workspaces/ourckpts/voxcpm/<이름>

    python tools/download_ckpts.py                 # 기본: VoxCPM2
    python tools/download_ckpts.py --models VoxCPM2 VoxCPM1.5
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path.home() / "workspaces/ourckpts/voxcpm"
REPOS = {"VoxCPM2": "openbmb/VoxCPM2", "VoxCPM1.5": "openbmb/VoxCPM1.5", "VoxCPM-0.5B": "openbmb/VoxCPM-0.5B"}

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["VoxCPM2"], choices=list(REPOS))
a = ap.parse_args()
for m in a.models:
    out = ROOT / m
    p = snapshot_download(REPOS[m], local_dir=str(out))
    n = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"[ok] {m} → {p} ({n/1e9:.2f} GB)")
