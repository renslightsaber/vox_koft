# 전체 FT 학습 안내 (voxko/sft)

> 📅 2026-11-05 · 처음 보는 사람용 · 코드 해설 [../reference/voxcpm2_training_code_walkthrough.md](../reference/voxcpm2_training_code_walkthrough.md) · LoRA 는 [lora_training.md](lora_training.md)
> 규칙: [../rules_in_this_proj/training_workflow.md](../rules_in_this_proj/training_workflow.md)

---

## 0. 한눈에
```bash
conda activate voxcpm && cd ~/workspaces/vox_koft
# smoke (3 step · 저장 안 함)
./scripts/run.sh train sft --exp configs/experiments/sft/<ID>.yaml --hw rtxpro6000_1gpu --gpu 0 --smoke
# 본 학습 (사용자가 byobu 에서)
./scripts/run.sh train sft --exp configs/experiments/sft/<ID>.yaml --hw rtxpro6000_4gpu
./scripts/run.sh train sft --exp configs/experiments/sft/<ID>.yaml --hw rtxpro6000_2gpu --accel 2gpu_a   # GPU 0,1
# 끊긴 뒤 이어서
./scripts/run.sh train sft --exp configs/experiments/sft/<ID>.yaml --hw rtxpro6000_4gpu --resume
# 여러 run 연속(실패해도 다음 · Ctrl-C 면 전체 중단 · 시작 전 GPU 확인) — 예: scripts/runs/20261106_A7N_4runs.sh
# TensorBoard(그룹 보기 · 포트 6008) — docs/guides/tensorboard_groups.md
./scripts/tb.sh
```

## 1. 설정 (3축 병합 + `--set`)
- `configs/config.yaml`(**스키마** · 기본값 · 모르는 키 에러) → `configs/hardware/rtxpro6000_<N>gpu.yaml`(batch · worker · GPU 수) → `configs/experiments/sft/<ID>.yaml`(이 run 만 다른 값) → `--set k=v` · `--max_steps N`.
- 상대경로: `model.*` 는 `~/workspaces/ourckpts/voxcpm/`, `data.*` 는 `~/workspaces/preprocessed/vox_koft/` 기준(환경변수 `VOX_CKPT_ROOT` · `VOX_DATA_ROOT` 로 바꿈).
- 자주 바꾸는 키

| 키 | 기본 | 뜻 |
|---|---|---|
| `data.emotion.map_file` | `configs/emotion/a7n_vox_v1.yaml` | 감정 → 영어 지시문 표 (학습 때 `"(지시문)문장"`) |
| `data.emotion.instruct_ratio` | 1.0 | 지시문 붙이는 비율 |
| `data.ref.ratio` · `pool` · `min_dur` · `max_dur` | 0.7 · neutral · 2 · 10 | 같은 화자 무감정 ref 를 붙이는 비율 · 후보 길이 |
| `data.max_seq_len` | 1024 | 근사 길이(text + target + ref 최대 + 4)가 넘으면 제외 |
| `data.batch_size` / `max_tokens` | hw yaml | GPU 당 micro-batch / 동적 배치 |
| `optim.lr` · `warmup_steps` · `scheduler` | 1e-5 · 100 · cosine_warmup | 공식 전체 FT 예 lr 1e-5 |
| `train.grad_accum` · `max_steps` | 2 · 1000 | 유효 배치 = batch × accum × GPU |
| `train.save_every` · `keep_last` · `resume.{every,keep}` | 250 · 12 · 500/1 | export 간격 · 보관 · 재개용 파일 간격/개수 |
| `train.lambdas.{diff,stop}` | 1 · 1 | loss 가중치 |

## 2. 무엇이 일어나나
1. **실행 정책**(모델을 만들기 전에): smoke / 새로 / 이어서 / 거부(같은 이름 run 이 있으면 `--resume` 또는 `--force_overwrite` 필요).
2. accelerate · GPU 확인(`VOX_EXPECT_GPUS` 와 실제 GPU UUID 대조) · seed 1986.
3. 모델: `VoxCPM2Model.from_local(training=True)`(fp32 master · autocast bf16) · 키 대조 · VAE 는 떼어 DDP 밖에.
4. 데이터: `VoxDataset` — 문장마다 `Random(seed|epoch|utt)` 로 지시문 · ref 를 결정(worker · GPU 수 · resume 과 무관하게 같음).
5. 루프: `BucketBatchSampler`(길이 버킷 · 창 배수) → packer(VAE encode) → forward → loss = Σ(λ·합계)/target 칸 수.
6. val(`eval_every`) · media(TB 합성) · save(`step_N/export_ft/` + 재개용 파일) · `_COMPLETE`.

## 3. 결과물
```
~/workspaces/outputs/vox_koft/ckpt/<run>/
  configs/   merged.yaml · 원본 yaml · emotion_map · accelerate · cmd.txt (🔒 수정 금지)
  logs/      <run>_<ts>.log  (run.sh 가 tee)
  step_N/    export_ft/{config.json, model.safetensors(bf16), audiovae.pth, tokenizer*, voxko_export.json}
             (+ 재개용: model.safetensors(fp32) · optimizer.bin · random_states_* · custom_checkpoint_*)
             _COMPLETE {"step", "epoch", "exports", "resume"}
~/workspaces/outputs/vox_koft/runs/<run>/   TensorBoard: train/* · val/* · val_unseen/* · mem/* · speed/* · media/*
```
- 합성: `python -m voxko.engine.synth --model <run>/step_N/export_ft --text "…" --emotion Happy-B --ref <무감정 wav> --out x.wav`

## 4. 실측 (정본 [reports/20261106_four_runs_preflight.md](../reports/20261106_four_runs_preflight.md))
- 가장 긴 배치(23초 + ref 10초): 1GPU bs4 70.4GB · bs6 84.6GB · bs8 OOM / **2GPU bs4 할당 81.9GB · 실제 점유 83GB**(DDP 로 +11.6GB).
- `scripts/run.sh` 는 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 를 기본으로 켭니다(실제 점유 89.9 → 85.1GB · loss 같음 · 2026-11-06).
- 속도(2GPU · 실제 데이터): 1.22~1.35초/step → 3000 step 약 66~72분(가설). 재개용 포함 저장 22초.
- 처음 smoke(11-05 · 짧은 문장 · batch 8): peak 71GB · 2.0~2.8초/step.
- 재개용 포함 step 약 31GB · export 4.6GB.
