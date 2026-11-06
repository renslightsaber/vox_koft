# `voxcpm` conda env 설치 가이드 (서버 112 · RTX PRO 6000 Blackwell)

> 📅 2026-11-05 · 처음 보는 사람용 · 이 문서대로 하면 VoxCPM2 를 이 서버에서 합성 · (나중에) 학습할 수 있습니다.
> 자매 프로젝트 `~/workspaces/q3tts_koft/requirements_112/README.md`(env `q3tts`)와 같은 틀입니다.

---

## 0. 한눈에

| 항목 | 값 |
|---|---|
| env 이름 | **`voxcpm`** |
| Python | 3.12.15 (q3tts 복제) — VoxCPM 요구 `>=3.10` |
| PyTorch | **2.8.0 + cu128** (Blackwell sm_120 에서 검증된 판) · torchaudio 2.8.0 · torchcodec 0.7.0 |
| 주요 패키지 | voxcpm 2.0.3(editable, `~/workspaces/vox_koft/VoxCPM`) · transformers 4.57.3 · funasr 1.4.21 · gradio 6.17 · modelscope 1.40 · wetext 0.2.1 · flash-attn 2.8.3 |
| 한국어 정규화 | `ko_tts_norm`(editable, `~/workspaces/vox_koft/ko-tts-text-normalizer`) |
| 설치 스크립트 | [`install_voxcpm_env.sh`](install_voxcpm_env.sh) |
| 검증 | [`verify_install.py`](verify_install.py) |
| 설치 결과 목록 | `requirements_112_voxcpm.lock.txt`(pip freeze) |
| 체크포인트 | `~/workspaces/ourckpts/voxcpm/VoxCPM2`(4.96GB · `../tools/download_ckpts.py`) |
| Jupyter 커널 | "Python (voxcpm)" |

## 1. 설치 — 한 줄
```bash
cd ~/workspaces/vox_koft
./requirements_112/install_voxcpm_env.sh 2>&1 | tee requirements_112/install.log
```
이 스크립트가 하는 일(순서대로):
1. `conda create -n voxcpm --clone q3tts` — **torch 를 새로 받지 않고** 서버에 이미 있는 q3tts env(Blackwell 검증 · torch 2.8.0+cu128 · flash-attn)를 복제합니다.
   - 🔴 **왜 복제?** 2026-11-05 서버 인터넷이 100~300KB/s 로 느려 torch(약 2.5GB)를 새로 받으면 몇 시간이 걸렸습니다. 복제는 1~2분.
2. q3tts 전용 editable 패키지(`qwen-tts` · `q3ko` · q3tts 쪽 `ko-tts-text-normalizer`)를 떼고, vox_koft 의 정규화기를 다시 연결합니다.
3. `torchcodec==0.7.*`(torch 2.8 짝) → `pip install -e VoxCPM`(나머지 의존성) → `hf_transfer` → Jupyter 커널 등록 → `pip freeze` 저장.

알려진 경고(무시해도 됨): `trl 1.16.0 requires datasets>=4.7.0, but you have datasets 3.6.0` — VoxCPM 이 `datasets<4` 를 요구해서 내려감. trl 은 q3tts 에서 복제돼 온 것이고 vox_koft 에서는 쓰지 않습니다.

## 2. 체크포인트 받기
```bash
~/miniforge3/envs/voxcpm/bin/python tools/download_ckpts.py              # VoxCPM2 (기본)
~/miniforge3/envs/voxcpm/bin/python tools/download_ckpts.py --models VoxCPM2 VoxCPM1.5
```
- 받는 곳: `~/workspaces/ourckpts/voxcpm/<이름>` · VoxCPM2 = `model.safetensors`(4.58GB) + `audiovae.pth`(0.38GB) + 토크나이저.
- 2026-11-05 실제 다운로드 약 7분.

## 3. 검증
```bash
CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/voxcpm/bin/python requirements_112/verify_install.py
```
2026-11-05 결과:
```
python 3.12.15 · torch 2.8.0+cu128 · CUDA 12.8 · GPU True · NVIDIA RTX PRO 6000 Blackwell Server Edition sm_120
transformers 4.57.3 · torchaudio 2.8.0+cu128 · ko_tts_norm OK
모델 로드 8.2s · 출력 SR 48000
입력: 밀크티로 공부하면서 더 궁금한 것이 있나요? 편하게 질문해 주세요.
합성 5.12s · 2.4s 걸림(RTF 0.46) · 최대 VRAM 5.9GB · 저장 ~/workspaces/vox_koft/_verify/verify_ko.wav
✅ verify OK
```
- RTF 0.46 은 **순수 PyTorch · torch.compile 끔(optimize=False) · 첫 합성 포함** 값입니다. 공식 수치(4090 · PyTorch 0.30 · Nano-vLLM 0.13)보다 느린 건 정상 — 서빙 속도는 따로 재야 합니다.
- ⚠️ **같은 seed 라도 모델을 올린 뒤 첫 합성만 값이 조금 다름**(GPU 준비) → 비교 · 재현이 필요하면 짧은 문장으로 한 번 "준비 합성"을 하고 시작합니다(2번째부터는 같은 seed = 같은 소리 확인).

## 4. 우리 패키지 voxko · 학습 검증 (2026-11-05)
```bash
conda activate voxcpm && cd ~/workspaces/vox_koft
pip install -e . --no-deps          # voxko (dependencies 는 비워 둠 — 이 폴더가 정본)
python -m pytest tests/             # CPU 39개 (설정 · 샘플러 · 체크포인트 · 데이터셋)
./scripts/run.sh train sft  --exp configs/experiments/sft/A7N100_ref_FT_00_smoke.yaml   --hw rtxpro6000_1gpu --gpu N --smoke
./scripts/run.sh train lora --exp configs/experiments/lora/A7N100_ref_LoRA_00_smoke.yaml --hw rtxpro6000_1gpu --gpu N --smoke
```
- 이 env 로 **처음 실제 학습**을 돌린 결과(전체 FT · LoRA 모두 ✅): 1 GPU smoke · resume 동일성 · 2 GPU · 저장물 합성 — [docs/reports/20261105_training_infra_smoke.md](../docs/reports/20261105_training_infra_smoke.md)

| | 전체 FT | LoRA r32 |
|---|---|---|
| smoke peak mem(batch 8 · 짧은 문장) | 71.0GB | 32.4GB |
| 속도(1 GPU) | 2.0~2.8초/step | 1.5~2.3초/step |
- 추가 설치 없음(accelerate 1.12.0 · tensorboard · soxr 1.1.0 · jiwer 가 이미 있음). 공식 학습 스크립트가 쓰는 `tensorboardX` 는 우리 트레이너가 쓰지 않습니다.
