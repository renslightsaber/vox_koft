# LoRA 학습 안내 (voxko/lora)

> 📅 2026-11-05 · 처음 보는 사람용 · 공통 내용은 [sft_training.md](sft_training.md) — 여기에는 **다른 점만**

---

## 0. 한눈에
```bash
./scripts/run.sh train lora --exp configs/experiments/lora/<ID>.yaml --hw rtxpro6000_1gpu --gpu 0 --smoke
./scripts/run.sh train lora --exp configs/experiments/lora/<ID>.yaml --hw rtxpro6000_2gpu --accel 2gpu_b
```
- 실험 yaml 에 **`lora.enable: true`** 가 있어야 합니다(없으면 학습 전 중단). 반대로 `train sft` 에 lora 가 켜져 있어도 중단.
- 데이터 · 지시문 · ref · resume · val 은 전체 FT 와 **똑같습니다**(같은 트레이너를 상속).

## 1. 설정
| 키 | 기본 | 뜻 |
|---|---|---|
| `lora.r` · `alpha` · `dropout` | 32 · 32 · 0 | 공식 예와 같음 |
| `lora.enable_lm` | true | base_lm · residual_lm 의 q/k/v/o |
| `lora.enable_dit` | true | DiT(feat_decoder.estimator) 의 q/k/v/o |
| `lora.enable_proj` | false | enc_to_lm · lm_to_dit · res_to_dit · fusion_concat |
| `optim.lr` | (실험 yaml 에서) 1e-4 | 공식 LoRA 예 |
- r32 · lm + dit → 학습 파라미터 18.1M(전체의 0.78%).

## 2. 결과물 · 불러오기
- `step_N/export_lora/lora_weights.safetensors`(키에 `lora_`, fp32 · 약 70MB) + `lora_config.json`
  - `{"base_model": <원본 경로>, "lora_config": {r, alpha, …}, "voxko": {run, step, "LoRAConfig": "V2"}}`
- 합성: `python -m voxko.engine.synth --model <run>/step_N/export_lora --text … --emotion … --ref … --out …`
  - base 는 `lora_config.json` 에서 읽습니다(바꾸려면 `--base`).
  - 🔴 V2 LoRAConfig 로 만들고, 불러온 개수가 모델의 LoRA 파라미터 수와 같은지 검사합니다. 공식 스크립트는 V1 클래스를 씁니다.
- 🔴 sm_120 에서 **nano-vllm 으로 LoRA 추론 금지**(공식 이슈 #326 — 소리가 나빠짐). 순수 PyTorch(`optimize=False`).

## 3. 실측 (2GPU 정본 [reports/20261106_four_runs_preflight.md](../reports/20261106_four_runs_preflight.md))
- 가장 긴 배치 2GPU bs8: 할당 54.2GB · 실제 점유 55.7GB · 속도 0.62~0.68초/step → 3000 step 약 34~37분(가설) · 재개용 포함 저장 8초.

### 3-1. 처음 smoke (2026-11-05 · smoke 데이터 · 1 GPU · batch 8)
- peak mem 32GB · 1.5~2.3초/step.
- 재개용 포함 step 8.9GB: accelerate `save_state` 가 frozen 가중치까지 모델 전체(fp32)를 저장하기 때문입니다. export 는 70MB 입니다.
