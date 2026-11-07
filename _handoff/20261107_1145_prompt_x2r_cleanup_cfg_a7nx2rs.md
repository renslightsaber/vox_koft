# vox_koft Claude 에게 — 청음 안내 · A7NX2R 재개용 파일 정리 · CFG 시험 · A7NX2Rs 학습 준비 (2026-11-07 11:45 KST · q3tts 세션 작성)

> 사용자는 깨어 있습니다. **학습은 사용자가 byobu 에서 직접 실행**합니다(이번엔 Claude 가 시작하지 않음). 준비 · smoke · 검증이 끝나면 실행 명령을 알려 주세요.
> 사용자는 A7NX2Rs 학습 중에 **SREF4(q3tts) · A7NX2R(vox) 노트북 청음**을 합니다 → 학습은 **GPU 2,3**, 노트북 · CFG 시험은 **GPU 0 또는 1** 을 씁니다(시작 전 `nvidia-smi` 확인).
> 결정이 필요한 갈림길은 먼저 사용자에게 묻습니다.

## 0. 배경(짧게)
- A7NX2R 두 판(lr 1e-4 · 5e-5)은 목소리가 지금까지 중 가장 좋지만 감정이 약해졌습니다(이 저장소 `docs/reports/20261107_A7NX2R_results.md`).
- q3tts 세션 의견: [docs/explanations/20261107_opinion_A7NX2R_A7NX3.md](../docs/explanations/20261107_opinion_A7NX2R_A7NX3.md)(복사본)
  - A7NX2R 은 목소리 ↔ 감정 맞바꿈 곡선을 넘은 것이 아니라 **목소리 쪽으로 옮겨 간 것**으로 보입니다(X2R02 s2000 ≈ LoRA01 s1500 ≈ A7NX3 s2250).
  - 주원인 추정: 무감정 비율 14.5% → 23.2%(감정 정답 연습 약 10% ↓).
- 사용자 결정(11/07 10~11시)
  1. 다음 데이터 = **A7NX2Rs**(무감정 정답의 ref 2 → 1 · 무감정 13.2%) · **lr 5e-5 한 판**
  2. **CFG 시험은 데이터와 따로**(재학습 없이)
  3. 부가: **감정 ↔ 목소리 그림** · **감정 ref 판 청음 페이지(작게)**
  4. A7NX2R 두 판 **재개용 파일 정리**

## 1. 사용자에게 알려 줄 것(채팅 답 맨 위에)
1. **청음 페이지 주소**(지금 있는 것 모두 · 열리는지 `curl` 로 확인)
   - D 재측정: http://127.0.0.1:8765/vox_koft/d_eval/index.html
   - 이번에 새로 만드는 감정 ref 판 · CFG 시험 페이지(§4 · §5)
2. **통합 평가(D 재측정) 기준 run 별 best step**(지금 보고서 기준 · 다시 확인해서 표로)

| run | best 5(참고 점수) |
|---|---|
| A7NX_xref_LoRA_01_r32 | 2250 · 1500 · 3000 · 750 · 500 |
| A7NX3_xref_LoRA_01_r32 | 4500 · 2250 · 9000 · 3000 · 1500 |
| A7NX2R_xref_LoRA_01_r32 | 4500 · 3000 · 1000 · 1500 · 2000 |
| A7NX2R_xref_LoRA_02_r32_lr5e5 | 4500 · 2000 · 3000 · 1500 · 1000 |

3. 노트북 경로(`notebooks/lora/…_listening.ipynb`)와 **GPU 0/1 로 여는 법**(노트북이 GPU 를 고르는 방법 한 줄)

## 2. A7NX2R 두 판 재개용 파일 정리(사용자 요청 · 승인됨)
- 대상: `~/workspaces/outputs/vox_koft/ckpt/{A7NX2R_xref_LoRA_01_r32, A7NX2R_xref_LoRA_02_r32_lr5e5}/step_*`
- **남길 것**: `export_lora/` · `_COMPLETE` · `configs/` · `logs/` · `_smoke/` — 노트북 청음 · 추론(합성)에 필요한 것 전부
- **지울 것**: 재개용 파일만(`model.safetensors` · `optimizer.bin` · `random_states_*.pkl` · `custom_checkpoint_*.pkl` 등) → `voxko.common.ckpt.remove_resume_files(step_dir, why)` · `_RESUME_REMOVED` 남김
- 지우기 **전**: 마지막 step 의 `export_lora` 로 짧은 합성 1개 확인 · 지운 **뒤**: 같은 합성 다시 1개
- 지운 용량 · 남은 파일을 인수인계에 적습니다. `export_*` 는 절대 지우지 않습니다.

## 3. A7NX2Rs 학습 준비(데이터는 q3tts 세션이 만들어 둠)
- 데이터: `~/workspaces/preprocessed/vox_koft/A7NX2Rs/`(`_READY` 있음) · 도구 `tools/make_a7nx2rs.py`(A7NX2R 도구 복제 + `K_NEU = 1`)
  - `train.jsonl` **203,110행** = A7N 191,355(감정 정답 88,200 × ref 2 + 무감정 정답 14,955 × ref 1) + 542 11,755(992명 × 약 12 × ref 1)
  - 무감정 비율 **13.2%**(A7NX2R 23.2%) · 542 비중 **5.8%**(10.2%)
  - **감정 행 176,400개는 A7NX2R 과 완전히 같음**(같은 정답 · 같은 ref) · 무감정 행은 A7NX2R 의 부분집합(코드로 확인)
  - 검사 7개 모두 0 · `val_xref.jsonl` = A7NX 것 그대로(192)
  - 설명: [docs/datasets/A7NX_vox_koft.md](../docs/datasets/A7NX_vox_koft.md) §A7NX2Rs
- 설정 1개: `configs/experiments/lora/A7NX2Rs_xref_LoRA_01_r32_lr5e5.yaml`
  - `A7NX2R_xref_LoRA_02_r32_lr5e5.yaml` 복제 → **데이터 경로(train · val_xref)와 머리 주석만** 바꿈
  - lr 5e-5 · LoRA r32 · cosine warmup 200 · 배치 8 × 2 GPU × accum 2 = 32 · **4,500 step** · eval/save 500 · keep_last 9 · resume {500, 1} · seed 1986
  - 4,500 step = 14.4만 행 ≈ 0.71 epoch(감정 정답은 약 1.4번 · 542 문장은 약 0.7번)
  - diff 로 A7NX2R 5e-5 판과 **데이터만 다름**을 확인해 보여 주세요.
- 이름 규칙(`docs/rules_in_this_proj/run_naming.md`)에 A7NX2Rs 추가 · `tools/tb_group.py` 그룹 확인
- **§3 학습 준비 — request_templates 대로**: 설정 검토 · smoke(오류 없이 `_COMPLETE` · 542 행 = Neutrality 지시문 + ref 1개 · 무감정 A7N 행 ref 1개 표본 출력 · 최대 메모리) · 학습 전 백업(`tools/pre_train_backup.py`)
- 그다음 **사용자에게 byobu 명령을 알려 주고 멈춥니다**(직접 실행하지 않음). 예:
```bash
cd ~/workspaces/vox_koft && conda activate voxcpm && \
./scripts/run.sh train lora --exp configs/experiments/lora/A7NX2Rs_xref_LoRA_01_r32_lr5e5.yaml --hw rtxpro6000_2gpu --accel 2gpu_b
```
- 예상 시간: 약 1시간(A7NX2R 5e-5 판 기준)

## 4. CFG 시험(재학습 없음 · GPU 0 또는 1 · 학습과 따로)
- 목적: 감정이 약할 때 **다시 학습하지 않고** 추론 CFG 로 감정을 올릴 수 있는지
- 대상: `A7NX2R_xref_LoRA_02_r32_lr5e5` **s2000 · s4500** × CFG **2.0(기준) · 2.5 · 3.0**
- 잣대: D 재측정(무감정 ref · 16명 × 7칸 × 2)과 같은 입력 · seed 1986 → e2v 6칸 · arousal · 목표 유사도 · 1위 · CER · 이상(끝 잘림 등)
- 먼저 확인: VoxCPM 의 CFG 가 **지시문에도 걸리는지**(코드 근거를 보고서에 한 줄). 걸리지 않으면 시험 의미가 작으니 그 사실을 먼저 알리고 사용자에게 진행 여부를 묻습니다.
- 작은 청음 페이지: 칸(기쁨B · 분노A · 분노B · 슬픔B) × 2명 × CFG 3개 · 버튼 판정 · CSV · **노트북 엔진과 같은 소리인지 재현 확인 뒤** 공개

## 5. 부가 작업
1. **감정 ↔ 목소리 그림**: 지금까지 모든 run · step(LoRA01 · A7NX · A7NX3 · A7NX2R 두 판 · CFG 시험 점 · 나중에 A7NX2Rs)을 (e2v 6칸, 목표 유사도) 점으로 · run 마다 선으로 이음 · 원본 표시 · 오차(약 ±3.6%p) 표시 → 보고서에 넣고, "곡선을 넘었는가"를 한 줄로
2. **감정 ref 판 청음 페이지(작게)**: unseen8 화자 × 기쁨 · 분노 ref × 지시문(무감정 · 다른 감정) 8줄 안팎 · LoRA01_s1500 vs A7NX_s1500 vs X2R01_s4500 vs X2R02_s4500 · 볼 것 = "ref 감정이 새어 나오는가" · 버튼 판정 · CSV · 재현 확인 뒤 공개

## 6. 학습이 끝나면(사용자가 알려 주면) — request_templates §1 대로
- 비교: LoRA01 · A7NX3 · A7NX2R 두 판 · A7NX2Rs
- 가장 궁금한 것
  1. 감정(e2v · arousal)이 A7NX3 수준으로 돌아왔는가
  2. 처음 보는 화자 목소리(1위 · B칸 이동) 이득이 남았는가
  3. 감정 ref 판 "중립 칸 중립"이 나빠지지 않았는가(무감정 정답 ref 가 1개라서)
  4. 감정 ↔ 목소리 그림에서 A7NX2Rs 의 위치
- 재개용 파일 정리는 **사용자 승인 뒤**

## 7. 지킬 것
- 한국어 · 처음 보는 사람용 문서 · 채팅 표는 좁게 · 시각은 KST(서버 + 9)
- **git commit / push 하지 않음**(사용자가 요청할 때만)
- 삭제는 §2 의 재개용 파일만 · `export_*` 절대 안 지움
- seed 1986 · bf16
- 끝나면 docs · 인수인계 업데이트(§2) · `_handoff/20261107_HHMM_*.md`
- q3tts 쪽 파일 · `Qwen3-TTS/` · `VoxCPM/` 은 수정하지 않음(데이터 `~/workspaces/preprocessed/vox_koft/A7NX2Rs/` 는 읽기만)
