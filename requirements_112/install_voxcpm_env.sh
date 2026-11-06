#!/usr/bin/env bash
# vox_koft — conda env "voxcpm" 설치 (서버 112 · RTX PRO 6000 Blackwell sm_120 · 드라이버 CUDA 13)
# 2026-11-05: 서버 인터넷이 느려(100~300KB/s) torch 를 새로 받지 않고, Blackwell 에서 검증된 q3tts env
#             (Python 3.12 · torch 2.8.0+cu128 · flash-attn 2.8.3)를 서버 안에서 복제한 뒤 VoxCPM 의존성만 더 설치한다.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -n voxcpm --clone q3tts
conda activate voxcpm
pip uninstall -y qwen-tts q3ko ko-tts-text-normalizer || true          # q3tts 쪽 editable 패키지 떼기
pip install -e ~/workspaces/vox_koft/ko-tts-text-normalizer --no-deps
pip install "torchcodec==0.7.*"                                         # torch 2.8 짝
pip install -e ~/workspaces/vox_koft/VoxCPM                             # 나머지 의존성(funasr · gradio · wetext …) — torch 는 이미 있음
pip install hf_transfer
python -m ipykernel install --user --name voxcpm --display-name "Python (voxcpm)"
pip freeze > ~/workspaces/vox_koft/requirements_112/requirements_112_voxcpm.lock.txt
echo "[install] 끝"
