# 학습 인프라 첫 검증 — 전체 FT · LoRA (smoke · resume · 2 GPU · 합성)

> 📅 2026-11-05(KST 15:30 전후) · 처음 보는 사람용 · 코드: `voxko/` · 해설: [../reference/voxcpm2_training_code_walkthrough.md](../reference/voxcpm2_training_code_walkthrough.md)
> 출력 · 로그: `~/workspaces/outputs/vox_koft/{ckpt/_smoke_logs, _verify/}`

---

## 0. 한눈에
- env `voxcpm` 에서 **처음으로 실제 학습**을 돌렸습니다. 전체 FT 와 LoRA 모두 아래 다섯 가지를 통과했습니다.

| 검사 | 전체 FT | LoRA r32 |
|---|---|---|
| CPU 단위 테스트(`pytest tests/`, 39개) | ✅ 공통 | ✅ 공통 |
| 1 GPU smoke(3 step · grad 검사 · val · 합성 · export 검증) | ✅ 최대 71.0GB | ✅ 최대 32.4GB |
| resume 동일성(step 4 저장 뒤 실제로 끊고 이어서) | ✅ 차이 = GPU 비결정성 크기 | ✅ 같음 |
| 2 GPU 짧은 실행(GPU 2,3 · 4 step · 저장 2회) | ✅ | ✅ |
| 저장물로 합성(처음 보는 화자 무감정 ref + 지시문) | ✅ 3.68초 | ✅ 3.04초 · LoRA 384개 모두 불러옴 |

- 데이터는 `A7N_smoke100`(화자 6명 · train 88 · val 12 · 처음 보는 화자 val 16 · 16kHz)을 썼습니다.
- 그때 GPU 4장 모두에서 사용자의 q3tts_koft SREF 합성 작업이 돌고 있었습니다(장당 약 7GB). 그래서 GPU 3(1장) · GPU 2,3(2장)을 함께 썼습니다.

## 1. 1 GPU smoke (GPU 3 · batch 8 · grad_accum 2)

| | 전체 FT | LoRA |
|---|---|---|
| 학습 파라미터 | 2,290M (100%) | 18.1M (0.78%) |
| step 1 · 2 · 3 loss | 1.0090 · 1.0496 · 0.9440 | 1.0090 · 1.0558 · 0.9506 |
| grad_norm (step 1) | 17.27 | 0.66 |
| 속도 | 2.0~2.8초/step | 1.5~2.3초/step |
| peak mem | 71.0GB | 32.4GB |
| val / val_unseen | 1.0813 / 0.9952 | 1.1072 / 1.0024 |

- step 1 loss 가 두 방식에서 같습니다(1.0090). LoRA B 는 0 으로 시작하므로 처음엔 원본과 같아야 맞습니다.
- `✓ [smoke] 모든 학습 대상 파라미터에 gradient 가 흐릅니다` — DDP unused-parameter 문제 없음.
- 정규화 분모 검사도 통과했습니다. collate 가 미리 센 target 칸 수와 packer 의 loss 칸 수가 같았습니다.
- 출처: `ckpt/_smoke_logs/A7N100_ref_{FT,LoRA}_00_smoke_*.log`

## 2. resume 동일성
방법:
1. 기준 run(6 step, 끊지 않음)을 돌립니다.
2. 같은 설정의 run 을 `step_4/_COMPLETE` 가 생긴 직후 **SIGTERM 으로 실제로 끊고** `--resume` 으로 6 step 까지 돌립니다.
3. TensorBoard 수치를 비교합니다(`_verify/resume_{sft,lora}/`).

| step | FT 차이(loss) | LoRA 차이(loss) | LoRA 기준 run 두 번끼리 차이 |
|---|---|---|---|
| 1~3 | 0 | 0 | 0 |
| 4 (끊기 전) | 3.1e-4 | 5.4e-4 | 2.8e-4 |
| 5 (resume 뒤) | 3.9e-4 | 2.9e-3 | 2.8e-3 |
| 6 | 8.7e-4 | 1.8e-4 | 1.0e-3 |

- lr · ref 비율(`train/ref_frac`)은 step 5 · 6 에서 **완전히 같습니다**. 스케줄과 데이터 위치("epoch 0 · 배치 8 부터")가 정확히 복원됐다는 뜻입니다.
- loss 의 작은 차이는 resume 때문이 아닙니다. 끊기 전(step 4)에도, 끊지 않은 기준 run 두 개끼리도 같은 크기로 납니다. GPU 역전파의 비결정성입니다.
- RNG 복원이 안 됐다면 CFM 잡음이 달라져 loss_diff 가 1e-2 이상 차이 났을 것입니다(가설 · 크기는 확인 안 함). 실제 loss_diff 차이는 1e-4 이하였습니다.

## 3. 2 GPU 짧은 실행 (`--hw rtxpro6000_2gpu --accel 2gpu_b` · 4 step)
- `GPU 확인: rank0 → nvidia-smi 2번 (지정 2,3)` · 유효 배치 ×4 · `Training End` · rc 0.
- val 의 평가 프레임 수가 1 GPU 와 같습니다(val 286 · val_unseen 673). **GPU 수와 무관하게 같은 표본**을 평가합니다.
- 저장 결과
  - step_2: export 만 남음. 재개용 파일은 keep=1 이라 지워지고 `_RESUME_REMOVED` 가 생김
  - step_4: 재개용 파일 + export + `_COMPLETE{"resume": true}`
- 크기

| | 재개용 포함 step | export |
|---|---|---|
| 전체 FT | 31GB | 4.6GB (bf16) |
| LoRA | 8.9GB | 70MB (fp32) |

- LoRA 의 재개용 파일이 큰 까닭: accelerate 가 frozen 가중치까지 모델 전체를 저장하기 때문입니다. 줄이려면 코드를 바꿔야 하고, 필요하면 나중에 합니다.
- 설정 스냅샷 `ckpt/<run>/configs/` 에는 config · hardware · experiment · accelerate · **emotion_map** · merged · cmd.txt 가 들어 있습니다(cmd.txt 에 실제 GPU 2,3 기록).

## 4. 저장물로 합성 (`voxko.engine.Engine` · GPU 3)
- 문장: `"오늘 시험에 합격했어요! 3년 동안 준비했거든요."`
  - ko_tts_norm 을 거쳐 `"(Very happy and excited, bright, cheerful and openly joyful)오늘 시험에 합격했어요! 삼년 동안 준비했거든요."` 가 됩니다.
- ref 는 처음 보는 화자(val_unseen8) 무감정 1개, seed 1986, cfg 2.0, timesteps 10 입니다.

| 모델 | 길이 | 같은 seed 두 번 | 끝 잘림 의심 | 비고 |
|---|---|---|---|---|
| 원본 VoxCPM2 | 3.04초 | 완전히 같음 | 없음 | |
| export_ft (4 step) | 3.68초 | 완전히 같음 | 없음 | 공식 `from_pretrained` 로 그대로 열림 |
| export_lora (4 step) | 3.04초 | 완전히 같음 | 없음 | V2 LoRAConfig · 384개 불러옴 · 건너뜀 0 |

- 4 step 짜리라 품질은 의미가 없습니다. **불러오기 · 합성 경로가 동작하는지**만 확인했습니다. wav 는 `_verify/synth/` 에 있습니다.

## 5. 실제 데이터 메모리 (전체 A7N 변환 뒤 · 가장 긴 32문장 21~23초 + ref 9~10초 강제 · 1 GPU smoke)
| | batch 4 | batch 6 | batch 8 |
|---|---|---|---|
| 전체 FT | **70.4GB** | 84.6GB | OOM |
| LoRA | — | — | **54.3GB** |
- 결정: 전체 FT = batch 4 × accum 4, LoRA = batch 8 × accum 2 → 둘 다 GPU 당 16문장/step.
- 본 학습 yaml 그대로 smoke(전체 데이터 106k 로드 포함 약 40초) ✅ · 2GPU 에서 **학습 도중 TB 합성(media) → 저장 → 학습 계속** ✅(FT peak 63.7GB · LoRA 25.1GB, TB 에 media/pred_audio_* 기록).
- ⚠️ 함께 도는 작업의 메모리는 변합니다(SREF 합성이 GPU 1 에서 8GB → 20.6GB 로 늘어 batch 8 smoke 가 OOM 난 적 있음) → 시작 전 `nvidia-smi`.

## 6. 남은 것 · 주의
- 공식 학습 스크립트(`train_voxcpm_finetune.py`) 자체는 돌리지 않았습니다. 같은 부품을 우리 트레이너로 돌렸습니다.
- `_verify/` 의 체크포인트는 사용자 승인으로 지웠습니다(로그 · TB · 합성 wav 만 남김).
