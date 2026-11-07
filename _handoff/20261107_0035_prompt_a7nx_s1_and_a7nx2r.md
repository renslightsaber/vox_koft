# vox_koft Claude 에게 — A7NX · A7NX3 정리 · §1 → A7NX2R LoRA 2판 준비 · 실행 · §1 (2026-11-07 00:35 KST · q3tts 세션 작성)

> 사용자는 이 지시문을 붙여 넣고 잠자리에 듭니다. **질문 없이 끝까지 진행**해 주세요. 결정이 필요한 갈림길은 아래 "정해진 것"을 따르고, 애매하면 안전한 쪽(삭제 · 덮어쓰기 안 함)을 고른 뒤 인수인계에 적어 주세요.
> **이번만 사용자 승인**: 검증이 끝나면 Claude 가 학습을 직접 시작합니다(nohup). byobu 명령도 문서에 남겨 주세요.

## 0. 배경(짧게)
- `A7NX_xref_LoRA_01_r32`(3,000 step) && `A7NX3_xref_LoRA_01_r32`(9,000 step)는 11/06 18:35 쯤 정상 종료했습니다.
- TB 를 보면 A7NX ≈ 기존 LoRA_01 입니다.
- A7NX3 는 **val_unseen(처음 보는 8명) 이 s750(0.8068) 바닥 → s1500 부터 오름 → s9000 0.8118** 입니다.
  - 같은 정답을 3번씩 보며 학습 화자 96명에 맞춰지는(외우는) 신호로 봅니다.
- 그래서 수정안 **A7NX2R** 을 만들었습니다(q3tts 세션 · 사용자 결정 11/07).
  - 정답마다 다른 칸 ref **2개**
  - **542 무감정 화자 992명 × 12문장**(같은 화자 다른 문장 ref 2개 · Neutrality 지시문)
  - 음질 나쁜 녹음 제외
- 설명 문서(q3tts 쪽 · 복사본이 이 저장소 `docs/datasets/A7NX_vox_koft.md` 에 붙을 예정): `~/workspaces/q3tts_koft/docs/datasets/A7NX_q3tts.md` §8 · `~/workspaces/q3tts_koft/docs/explanations/20261107_why_mix_542.md`

## 1. A7NX · A7NX3 재개용 파일 정리(두 run · 학습 끝남 · 사용자 요청)
- 대상: `~/workspaces/outputs/vox_koft/ckpt/{A7NX_xref_LoRA_01_r32, A7NX3_xref_LoRA_01_r32}/step_*`
- **남길 것**: `export_lora/` · `_COMPLETE` · `configs/` · `logs/` · `_smoke/`
- **지울 것**: 재개용 파일만(`model.safetensors` · `optimizer.bin` · `random_states_*.pkl` · `custom_checkpoint_*.pkl` 등) → `voxko.common.ckpt.remove_resume_files(step_dir, why)` 를 쓰고 `_RESUME_REMOVED` 를 남깁니다.
- 지우기 **전에**: 마지막 step 하나에서 `export_lora` 로 짧은 합성 1개가 되는지 확인합니다(청음 · 추론에 필요한 것이 남는지).
- 지운 용량과 남은 파일 목록을 인수인계에 적습니다. `export_*` 는 절대 지우지 않습니다.

## 2. 두 run — request_templates §1 대로
- 종료 확인 → 학습 뒤 백업(`tools/post_train_backup.py`) → 분석 보고서 → 청음 노트북 + best 5 → 통합 평가(D 재측정)
- GPU: **2,3** 을 씁니다(q3tts 세션이 0,1 에서 SREF v4 데이터 합성 · 학습을 합니다). 시작 전에 `nvidia-smi` 로 확인합니다.
- 볼 step
  - A7NX: 750 · 1500 · 2250 · 3000
  - **A7NX3: 750 · 1500 · 2250 · 3000 · 4500 중심**(외우기 전후) · 9000 은 비교로 1개
- 꼭 넣을 것
  1. 칸별 **"생성 중립 대비" 목소리 이동**(오늘 분석 `docs/reports/20261106_cell_voice_drift_analysis.md` 와 같은 잣대) — 기존 LoRA_01 vs A7NX vs A7NX3. 특히 Happy-B · Angry-B · Angry-A
  2. **감정 ref 판 D 재측정**(목표 화자의 감정 녹음을 ref 로 주고 다른 칸 만들기 · 감정이 ref 에서 새어 나오는지)
  3. **이전 run(LoRA_01 등)의 val_xref 다시 재기**(같은 `A7NX/val_xref.jsonl` 192)
  4. 무감정 ref 판 D 재측정(기존 기준선과 같은 잣대)
- seed 1986 · 준비 합성 뒤 재현

## 3. A7NX2R 학습 준비(데이터는 이미 있음)
- 데이터: `~/workspaces/preprocessed/vox_koft/A7NX2R/`(`_READY` 있음)
  - `train.jsonl` 229,820행 = A7N 206,310(정답 103,155 × ref 2) + 542 23,510(정답 11,755 × ref 2)
  - 화자: A7N 96 + 542 992
  - 무감정 비율 23.2%(A7NX 14.5%) · 542 비중 10.2%
  - 행마다 고정 ref(`ref_audio` · `ref_utt` · `ref_duration` · `utt_x`) → 지금 dataset 의 "고정 ref" 경로가 그대로 받습니다.
  - 542 행은 `emotion: Neutrality`(지시문 표의 Neutrality 문구가 붙음) · `src: "542"` 칸으로 구분
  - `val_xref.jsonl` = A7NX 것 그대로(192 · 이전 run 과 같은 잣대)
  - 검사: 같은 칸 ref · 자기 자신 · 다른 화자 ref · 음질 걸림 · 처음 보는 화자 누출 · 중복 = 모두 0
  - 만든 도구: `tools/make_a7nx2r.py`
- 설정 2개(`configs/experiments/lora/`): 두 판은 **lr 만 다름**

| run | lr |
|---|---|
| `A7NX2R_xref_LoRA_01_r32` | 1e-4 |
| `A7NX2R_xref_LoRA_02_r32_lr5e5` | 5e-5 |

- 공통 설정(나머지는 `A7NX3_xref_LoRA_01_r32.yaml` 복제)
  - `train_jsonl: A7NX2R/train.jsonl` · `val_xref_jsonl: A7NX2R/val_xref.jsonl` · val · val_unseen · ref pool 은 A7NX3 와 같음
  - LoRA r32 · alpha 32 · cosine_warmup · warmup 200 · batch 8 × 2 GPU × accum 2 = 32
  - **max_steps 4,500**(약 14.4만 행 ≈ 정답당 약 1.2번 · A7NX3 의 val_unseen 이 오르기 시작한 지점 근처에서 멈춤)
  - eval_every 500 · save_every 500 · keep_last 9 · resume {every 500, keep 1} · seed 1986
- 이름 규칙(`docs/rules_in_this_proj/run_naming.md`)에 A7NX2R 을 추가하고, `tools/tb_group.py` 그룹도 확인합니다.
- 확인할 것(smoke)
  - 542 행이 Neutrality 지시문 + 고정 ref 로 들어가는지(표본 몇 개 입력 문자열 · ref_utt 출력)
  - smoke 가 오류 없이 `_COMPLETE`
  - 학습 전 백업(`tools/pre_train_backup.py`)
  - TB 그룹 위치
- ⚠️ 메모리: 542 행은 A7N 보다 짧아 문제는 없을 것으로 보지만, smoke 에서 최대 메모리를 적어 둡니다.

## 4. 실행(이번만 Claude 가 · §2 평가로 GPU 2,3 이 비면)
- 아래 명령을 `nohup` 으로 실행하고 로그 경로를 인수인계에 적습니다. 같은 명령(byobu 용)도 문서에 남깁니다.
```bash
cd ~/workspaces/vox_koft && conda activate voxcpm && \
./scripts/run.sh train lora --exp configs/experiments/lora/A7NX2R_xref_LoRA_01_r32.yaml --hw rtxpro6000_2gpu --accel 2gpu_b && \
./scripts/run.sh train lora --exp configs/experiments/lora/A7NX2R_xref_LoRA_02_r32_lr5e5.yaml --hw rtxpro6000_2gpu --accel 2gpu_b
```
- (`--accel 2gpu_b` 가 GPU 2,3 인지 먼저 확인합니다. 아니면 맞는 것으로 바꿉니다.)
- 예상 시간: 한 판 약 45~50분(A7NX3 9,000 step ≈ 1.5시간 기준) · 둘이면 약 1.6시간

## 5. 학습이 끝나면 — request_templates §1 대로
- 비교 대상: 기존 LoRA_01 · A7NX · A7NX3(앞쪽 step) · A7NX2R 두 판
- 가장 궁금한 것
  1. **val_unseen 이 A7NX3 처럼 오르지 않는가**(542 효과)
  2. 칸별 목소리 이동이 줄었는가(특히 B 칸)
  3. 감정이 약해지지 않았는가(무감정 비율 14.5 → 23%)
  4. lr 1e-4 vs 5e-5
- 재개용 파일 정리는 **사용자 승인 뒤**에 합니다(이번 두 판).

## 6. 지킬 것
- 한국어 · 처음 보는 사람용 문서 · 시각은 KST(서버 + 9)
- **git commit / push 하지 않음**
- 삭제는 §1 의 재개용 파일만
- seed 1986 · bf16
- 끝나면 docs · 인수인계 업데이트(§2) · `_handoff/20261107_HHMM_*.md`
- q3tts 쪽 파일 · `Qwen3-TTS/` · `VoxCPM/` 은 수정하지 않습니다(데이터 `~/workspaces/preprocessed/vox_koft/A7NX2R/` 는 읽기만).
