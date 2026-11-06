# VoxCPM2 공식 학습 코드 해설 — 그리고 우리(voxko)가 바꾼 점

> 📅 2026-11-05 · 처음 보는 사람용 · 공식 clone `VoxCPM/`(커밋 cce58a5, 🔒 수정 금지) 기준
> 관련: [voxcpm2_overview_finetune.md](voxcpm2_overview_finetune.md)(조사) · [../guides/sft_training.md](../guides/sft_training.md) · [../guides/lora_training.md](../guides/lora_training.md)

---

## 0. 한눈에
- 공식 FT 는 `VoxCPM/scripts/train_voxcpm_finetune.py` 하나입니다. 실행하면 아래 순서로 돕니다.
  1. JSONL(`audio` · `text` · 선택 `ref_audio`)을 HF datasets 로 읽습니다.
  2. GPU 에서 AudioVAE 로 소리를 latent 로 바꿉니다(16kHz 입력).
  3. LM + DiT 에 넣어 **loss/diff(흐름 정합 MSE) + loss/stop(끝 판단 CE)** 를 계산합니다.
  4. AdamW + cosine 으로 학습하고, `step_N/` 에 저장합니다.
- 괄호 지시문 `"(감정 묘사)문장"` 은 **평범한 텍스트**입니다. 특별 처리가 없으므로 학습 text 에 그대로 넣으면 됩니다.
- 공식 코드는 자체 DDP 를 쓰고, resume 때 **데이터 위치 · RNG 를 복원하지 않습니다**. 그 밖에도 우리 환경과 맞지 않는 점이 있어서, **우리 트레이너(`voxko`, HF accelerate)에 공식 부품(모델 · packer · LoRA)을 끼워** 씁니다(§5).

## 1. 데이터 → 배치

| 단계 | 공식 코드 | 내용 |
|---|---|---|
| 읽기 | `training/data.py load_audio_text_datasets` | JSONL → HF datasets. `Audio(sampling_rate=16000)` 이라 **읽을 때 16kHz 로 리샘플** |
| 토큰화 | `train_voxcpm_finetune.py:116-153` | `model.text_tokenizer`(Llama 토크나이저 + 중국어 여러 글자 토큰을 한 글자씩). **BOS/EOS 없음** |
| 길이 거르기 | `:164-182` | `max_batch_tokens // batch_size` 보다 긴 문장 제외(train 만, `sf.info` 로 길이 읽음) |
| 묶기 | `training/data.py HFVoxCPMDataset.collate_fn` | text 는 -100, 파형은 -100.0 으로 오른쪽 패딩 |
| 배치 구성 | `training/packers.py AudioFeatureProcessingPacker` | 한 문장씩 VAE encode(GPU · fp32 · no_grad) → 아래 모양 |

학습 입력 모양(`packers.py:249-441`)
```
ref 없음:  [ text…, 101, target_latent…, 102 ]
ref 있음:  [ 103, ref_latent…, 104, text…, 101, target_latent…, 102 ]   ← 추론 reference_wav_path 와 같은 모양
loss:      target_latent 칸에만 (ref · text · 특수 토큰은 0)
stop 라벨: 마지막 target 칸 = 1
```
- latent 한 칸(patch) = AudioVAE hop 640 × patch 4 = 2560 샘플 = **0.16초**(LM 6.25 step/초).
- 🔴 배치 길이가 `config.max_length`(8192)를 넘으면 **말없이 잘립니다** → stop 라벨이 사라집니다.

## 2. 모델 · loss (`model/voxcpm2.py:302-396`)
- `VoxCPM2Model.forward(text_tokens, text_mask, audio_feats, audio_mask, loss_mask, position_ids, labels, progress=…)`
  - 텍스트 임베딩과 latent 임베딩(feat_encoder → enc_to_lm_proj)을 mask 로 합칩니다 → base_lm(MiniCPM, causal) → FSQ.
  - residual_lm → lm_to_dit / res_to_dit → DiT(`feat_decoder`)
- **loss/diff**: `unified_cfm.compute_loss`
  - CFM 흐름 정합 MSE 를 target 칸 평균으로 냅니다.
  - `progress = step/전체` 가 잡음 조건 확률과 mean-mode 비율을 바꿉니다.
  - `training_cfg_rate` 로 조건을 무작위로 지워 CFG 를 학습합니다.
- **loss/stop**: `stop_head` CE 를 target 칸 평균으로 냅니다.
- 전체 loss = λ_diff·diff + λ_stop·stop(공식 기본 1 : 1).
- `from_local(training=True)` 는 모델을 **fp32** 로 둡니다(학습 때 autocast bf16). VAE 는 freeze.

## 3. 저장 · 불러오기

| | 공식 저장(`save_checkpoint`) | 공식 불러오기 |
|---|---|---|
| 전체 FT | `step_N/model.safetensors`(audio_vae 제외, fp32) + config · audiovae · tokenizer 복사 | `VoxCPM.from_pretrained(step_N)` |
| LoRA | `step_N/lora_weights.safetensors`(키에 `lora_`) + `lora_config.json {base_model, lora_config}` | `VoxCPM.from_pretrained(base, lora_config=…, lora_weights_path=step_N)` |
| 공통 | `optimizer.pth` · `scheduler.pth` · `training_state.json{step}` · `latest/`(통째 복사) | resume 은 `latest/` 에서만 |

- LoRA(`voxcpm2.py:138-277`)
  - 대상: `enable_lm` = base_lm · residual_lm, `enable_dit` = feat_decoder.estimator. 둘 다 q/k/v/o_proj 를 바꿉니다.
  - `enable_proj` 는 enc_to_lm · lm_to_dit · res_to_dit · fusion_concat 입니다.
  - `LoRALinear` 는 원래 weight 키를 그대로 두고 `lora_A`(kaiming) · `lora_B`(0) 를 더합니다. scaling = alpha/r 입니다.

## 4. 공식 코드의 문제 · 우리 환경과 안 맞는 점 → 우리 해결

| # | 공식 코드 | 문제 | 우리(voxko) |
|---|---|---|---|
| 1 | 자체 `training/accelerator.py`(torch DDP · torchrun) | q3tts_koft 인프라(accelerate · 설정 3축)와 다름 | **HF accelerate** + `scripts/run.sh` 의 hw ↔ accelerate 짝 |
| 2 | resume = weight · optimizer · scheduler · step 만 | **데이터 위치 · sampler epoch · RNG · GradScaler 없음**. step 하나가 어긋남 | `accelerate save_state` + `TrainState(epoch · batch_in_epoch · sched · seed)` → **정확히 다음 배치부터**(실측: 끊지 않은 run 과 같음, §6) |
| 3 | `latest/` 를 rmtree 뒤 copytree · 보관 개수 제한 없음 · step 0 저장 | 원자적이지 않음 · 디스크 | `train.resume {every, keep}` · `keep_last` 회전 · export 는 `.tmp` → rename · `_COMPLETE` |
| 4 | train 지표 = rank0 의 마지막 micro-batch | 평균이 아님 | 창 · 전 GPU **합계 / target 칸 수**(모든 칸 같은 가중치) |
| 5 | val: rank 마다 배치 수가 다르면 all_reduce hang · rank0 혼자 생성(NCCL timeout 위험) · step 0 에도 val | 멈춤 위험 | DDP 를 거치지 않는 forward + 합계 all-reduce · **GPU 수와 무관한 같은 표본**(EvalBatchSampler) |
| 6 | `--config_path` 를 주면 다른 CLI 인자 무시 | 실수 위험 | 스키마 검사(모르는 키 에러) + `--set k=v` |
| 7 | `max_length` 넘으면 말없이 자름 | stop 라벨 사라짐 | `data.max_seq_len` 으로 **미리 제외**하고 개수를 배너에 표시 |
| 8 | `from_local` 이 `strict=False` 로 조용히 불러옴 | 키가 빠져도 모름 | 키 이름으로 직접 대조(빠지면 중단) |
| 9 | `validate.py` 는 16kHz 아니면 에러, `data.py` 는 리샘플 | 서로 다름 | **미리 16kHz wav 로 변환**(`voxko.data.prepare`) + Dataset 이 16kHz 아니면 에러 |
| 10 | 공식 추론 스크립트 · `core.py` 가 **V1 LoRAConfig** import | `fusion_concat_proj` 없음(proj 켤 때 차이) | engine 은 **V2** `voxcpm.model.voxcpm2.LoRAConfig` |
| 11 | `tensorboardX` 가 공식 의존성에 없음 | import 오류 가능 | accelerate 의 TensorBoard 사용 |
| 12 | GradScaler 를 bf16 에도 씀 | 불필요 | accelerate bf16(스케일러 없음) |
| 13 | 학습 text 공백을 정리하지 않음(추론은 `\s+`→` `) | 학습 ≠ 추론 | Dataset 이 추론과 같게 공백 정리 |
| 14 | sm_120 + LoRA + nano-vllm 소리 이상(공식 이슈 #326) | 검증 왜곡 | 학습 검증 · engine 은 **순수 PyTorch**(`optimize=False`) |

## 5. 우리 구조 (공식 부품을 어디에 끼웠나)

```
scripts/run.sh train sft|lora
  └ voxko/{sft,lora}/train.py         설정 3축 병합(voxko.common.config) · ConfigError → rc 2
     └ VoxSFTTrainer / VoxLoRATrainer   (voxko/sft/trainer.py · voxko/lora/trainer.py)
        ├ BaseTrainer                  voxko/common/trainer_base.py (q3tts_koft 판 이식: 정책 · 루프 · val · save · resume)
        ├ build_model                  ★ VoxCPM2Model.from_local(training=True, lora_config=V2) · VAE 를 떼어 DDP 밖에
        ├ build_datasets               voxko/data/dataset.py VoxDataset — "(지시문)문장" · 무감정 ref 를 결정적으로 고름
        ├ forward_step                 ★ AudioFeatureProcessingPacker(fp32) → ★ model.forward → diff · stop 합계
        └ export                       step_N/export_ft/ (공식 추론 형식) · step_N/export_lora/ (lora_weights + lora_config)
voxko/engine/synth.py                  ★ VoxCPM.from_pretrained(원본 | export_ft | base+export_lora) · ko_tts_norm · 준비 합성
```
★ = 공식 코드를 import 해서 그대로 씀.

## 5-1. D안과 같은 모양인가
- 학습 ref 경로 `[103, ref, 104, text, 101, 정답, 102]` 의 앞부분이 추론 "reference 만" 경로(`voxcpm2.py:537-575`) `[103, ref, 104, text, 101]` 와 같습니다.
- prompt 이어 말하기 모양(`is_prompt`)은 쓰지 않습니다 → [plan §2](../plan/20261105_success_criteria_and_next_runs.md).

## 6. 검증 (2026-11-05 · 결과 정본 [../reports/20261105_training_infra_smoke.md](../reports/20261105_training_infra_smoke.md))
- smoke · resume 동일성 · 2 GPU · 저장물 합성을 전체 FT 와 LoRA 둘 다 확인했습니다.
- 공식 학습 스크립트 자체는 돌리지 않았습니다. 우리 트레이너가 같은 부품(모델 forward · packer)을 그대로 쓰기 때문입니다.
