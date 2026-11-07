# vox_koft Claude 에게 — A7NX2Rs · A7NX2Rs6 두 판: 준비 → 학습(직접 실행) → §1 끝까지 (2026-11-07 15:30 KST · q3tts 세션 작성)

> 오늘은 주말이고, 사용자는 서버를 켜 둔 채 자리를 비울 수 있습니다. 그래서 **어젯밤처럼 이번에도 vox Claude 가 학습을 직접 시작하고(nohup), 끝나면 §1 까지 이어서** 합니다(사용자 승인 11/07 15:15).
> 결정이 필요한 갈림길이 생기면 그 단계에서 멈추고 사용자에게 묻습니다. 단, 아래에 정해진 것은 묻지 않고 진행합니다.
> 지난 지시문: `_handoff/20261107_1145_prompt_x2r_cleanup_cfg_a7nx2rs.md` · 지난 인수인계: `_handoff/20261107_1215_handoff_main.md`

## 0. 무엇이 바뀌었나(짧게)
**사용자 청음**(D 재측정 · X2R02lr5e5_s4500)
- 목소리 유지는 전보다 훨씬 좋아졌습니다.
- 542 화자 일부는 감정이 약해졌습니다.
  - a542_2214_JCE: 기쁨B · 분노A/B 가 약하고, 슬픔 칸도 반만 감정이 납니다.
  - a542_2108_KKH: 분노B 1문장은 감정은 맞지만 목소리가 다릅니다.
- a542_1837_KHG 는 7칸 모두 무난했습니다.
- unseen8 aihub_0028 은 기쁨B · 분노B 에서 목소리가 다르게 들렸습니다.

**q3tts 세션의 분해(D 재측정 e2v 맞힘 · 화자 묶음별)**

| 모델 | 542 A칸 | 542 B칸 | unseen8 A | unseen8 B |
|---|---|---|---|---|
| LoRA01 s1500 | 42% | 54% | 52% | 60% |
| A7NX3 s2250 | 44% | 48% | 54% | 62% |
| X2R02 s2000 | 29% | 56% | 65% | 65% |
| X2R02 s4500 | **19%** | 58% | 60% | 65% |

- **감정 손실은 542 화자에 몰려 있습니다.** 542 화자는 무감정 낭독만 녹음한 사람입니다. unseen8 은 오히려 올랐습니다.
- 가설: 학습의 542 행("낭독 목소리 ref → 늘 무감정 정답")이 **"낭독 말투 목소리 = 무감정" 지름길**을 가르쳤습니다.

**사용자 결정(11/07 15:15)**
1. **A7NX2Rs 그대로 학습** + **542 를 더 줄인 판 A7NX2Rs6** 을 이어서(&&) 학습합니다. 두 판은 **542 양만 다릅니다.** lr 은 바꾸지 않습니다.
2. unseen8 은 **평가용으로 유지**합니다(학습에 넣지 않음).
3. d_eval 청음 페이지에 **unseen8 진짜 감정 녹음 열**을 추가합니다(있는 칸만).
4. (q3tts 쪽 · 참고) 71349 의 안 쓴 화자 26명 말투 청음 페이지는 q3tts 가 만들었습니다. vox 가 할 일은 없습니다.

## 1. 사용자에게 알려 줄 것(채팅 답 맨 위)
1. 청음 페이지(열리는지 `curl` 로 확인)
   - D 재측정: http://127.0.0.1:8765/vox_koft/d_eval/index.html(+ 진짜 녹음 열 · §5)
   - CFG: http://127.0.0.1:8765/vox_koft/cfg_test/index.html
   - 감정 ref 판: http://127.0.0.1:8765/vox_koft/d_eval_eref/index.html
   - (q3tts) 26명 말투: http://127.0.0.1:8765/datasets/a71349_unused26/index.html
2. 학습 시작 시각 · 예상 끝 시각(KST) · 실행 중인 GPU
3. 두 run 이 끝나면 §1 결과 요약: 핵심 표 = 542 vs unseen8 · best 5

## 2. GPU 규칙(겹치지 않게)
| GPU | 누가 |
|---|---|
| **2,3** | vox 학습 · 평가(이 지시문 전부) |
| **0** | q3tts 세션의 x-vector 감정 방향 실험(오늘 15:30~ · 쓰지 마세요) |
| **1** | 사용자 청음 노트북(q3tts · vox) |

- 시작 전에 `nvidia-smi` 를 확인합니다.
- 사용자 vox 노트북 커널이 GPU 2 에 있으면, 사용자에게 "GPU 1 로 옮겨 주세요"라고 알립니다.
  - 메모리는 충분하므로(smoke 42.6GB + 12.6GB) 학습은 기다리지 않고 시작합니다.

## 3. 학습 준비(§3) — 2 run
**A. `A7NX2Rs_xref_LoRA_01_r32_lr5e5`**
- 이미 준비되어 있습니다(설정 · smoke · 백업 · 12:15 인수인계 §3). 바뀐 것이 없는지 백업과 diff 만 확인합니다.

**B. 🆕 `A7NX2Rs6_xref_LoRA_01_r32_lr5e5`**

데이터(q3tts 세션 작성 · 읽기만): `~/workspaces/preprocessed/vox_koft/A7NX2Rs6/`(`_READY` 있음)
- 도구: `tools/make_a7nx2rs6.py` = make_a7nx2rs.py 복제 + **N542 = 6**
- `train.jsonl` **197,296행**
  - A7N 191,355 — **A7NX2Rs 와 완전히 같음**
  - 542 5,941 — 992명 × 약 6 · ref 1 · **A7NX2Rs 542 행의 부분집합**
- 542 비중 **3.0%**(A7NX2Rs 5.8%)
- 무감정 **10.6%**(A7NX2Rs 13.2%) — 542 가 무감정이라 함께 줄어듭니다. 보고서에 꼭 적습니다.
- 검사 7개는 모두 0입니다. `val_xref.jsonl` 은 A7NX 것 그대로(192)입니다.

설정: A7NX2Rs 설정 복제 → **데이터 경로(train · val_xref) · run_name · 머리 주석만** 바꿉니다.
- lr 5e-5 · r32 · 배치 32 · **4,500 step**(두 판 같은 step 이라 비교가 쉬움) · eval/save 500 · keep_last 9 · resume {500, 1} · seed 1986
- diff 로 A7NX2Rs 판과 **데이터만 다름**을 보여 줍니다.

그 밖에
- 이름 규칙 `docs/rules_in_this_proj/run_naming.md` 에 A7NX2Rs6 을 추가합니다.
- `tools/tb_group.py` 그룹을 확인합니다(`LoRA/A7NX2Rs6/…`).
- 평가 도구 이름표 `X2Rs6lr5e5` 를 추가합니다: eval_runs · build_listening_nb · d_eval_tables · d_eval_eref · emo_voice_plot
- smoke
  - 오류 없이 `_COMPLETE`
  - 542 행 = Neutrality 지시문 + ref 1개
  - 최대 메모리
- 학습 전 백업: `tools/pre_train_backup.py`

## 4. 학습(vox Claude 가 직접 · nohup · GPU 2,3)
- 두 판을 이어서 돌립니다(약 2시간).
```bash
cd ~/workspaces/vox_koft && conda activate voxcpm && \
./scripts/run.sh train lora --exp configs/experiments/lora/A7NX2Rs_xref_LoRA_01_r32_lr5e5.yaml --hw rtxpro6000_2gpu --accel 2gpu_b && \
./scripts/run.sh train lora --exp configs/experiments/lora/A7NX2Rs6_xref_LoRA_01_r32_lr5e5.yaml --hw rtxpro6000_2gpu --accel 2gpu_b
```
- 같은 명령을 인수인계에 **byobu 용으로도** 남깁니다.
- 진행은 로그 · `_COMPLETE` 로 확인합니다.
- 첫 run 이 실패하면 두 번째는 시작하지 않고, 원인을 정리해 사용자에게 묻습니다.

## 5. d_eval 진짜 녹음 열(학습 중에 · GPU 쓰지 않음)
- 대상: unseen8 8명 × 7칸. **진짜 녹음이 있는 칸만** 1개씩 넣습니다.
- 고르는 순서
  1. `~/workspaces/preprocessed/q3tts_koft/q3_extra/val_unseen8/val.jsonl`(24kHz · 칸 이름 그대로)
  2. 없으면 `~/workspaces/ourdata/aihub_71349/meta/metadata.csv`
     - 화자 = reciter_id · 강도 1 → A · 3 → B · 투표 4.5↑ · 2~10초
     - 원본 = `/data1/rohoon/aihub_71349/<audio_path>`
  3. 그래도 없으면 "없음"
- 조건
  - **평가 문장 · ref 녹음과 겹치지 않게** 합니다.
  - seed 1986 으로 고릅니다.
  - 열 머리에 "진짜 사람 녹음(비교용 · 말투 · 문장 다름)"을 적습니다.
- 참고: unseen8 의 B칸 진짜 녹음은 거의 없습니다(강도 3 은 aihub_0012 의 분노 · 슬픔뿐).
- 기존 판정 버튼 · CSV 형식은 그대로 둡니다. 판정 localStorage 키를 바꾸지 마세요(사용자 판정이 남아 있을 수 있음).
- 원본 녹음이라 재현 확인은 필요 없습니다. 칸 · 화자가 맞는지만 표본으로 확인합니다.

## 6. 학습이 끝나면 — request_templates §1 대로(묻지 않고 진행)
1. 종료 확인(로그 · `_COMPLETE` 9개씩 · 오류 · 재개용 파일은 마지막 1개) · 학습 뒤 백업
2. 보고서 `docs/reports/20261107_A7NX2Rs_results.md`
   - 비교: LoRA01 · A7NX3 · X2R02(5e-5) · X2Rs01 · X2Rs6
   - TB(val_unseen · val_xref)
3. 청음 노트북 2개 + best 5 주석
4. **통합 평가 = D 재측정**(두 run · 9 step · 무감정 ref 16명 × 7칸 × 2 · seed 1986)
   - **핵심 표: 542 8명 vs unseen8 8명 나눠 보기** — A/B칸 e2v · arousal · 목표 유사도 · 1위 · CER · 이상
   - 화자별 e2v 맞힘(12개 중)도 부록으로
   - 가장 궁금한 것
     1. 542 A칸 감정이 X2R02 19% → LoRA01 수준(42%)으로 돌아오는가
     2. A7NX2Rs6(542 3%)이 A7NX2Rs(5.8%)보다 더 나은가 = 지름길 가설
     3. unseen8 감정 · 목소리 이득(1위 · B칸 이동)이 남았는가
5. 감정 ↔ 목소리 그림 갱신: 두 판 추가 · **542 / unseen8 따로 그린 작은 그림 2장** 추가
6. 감정 ref 판(d_eval_eref)에 X2Rs01 · X2Rs6 마지막 step 열 추가 · 재현 확인 뒤 공개
7. best step 표(run 별 best 5) · 채팅 요약 · docs · 인수인계(§2)
8. **재개용 파일 정리는 하지 않습니다**(사용자 승인 뒤)

## 7. 지킬 것
- 한국어 · 처음 보는 사람용 문서 · 채팅 표는 좁게 · 시각은 KST(서버 + 9)
- **git commit / push 하지 않음**
- 삭제하지 않음(`export_*` 절대 안 지움)
- seed 1986 · bf16 · GPU 는 2,3 만
- q3tts 쪽 파일 · `Qwen3-TTS/` · `VoxCPM/` 은 수정하지 않습니다. 데이터 `~/workspaces/preprocessed/vox_koft/A7NX2Rs*/` 는 읽기만 합니다.
- 끝나면 `_handoff/20261107_HHMM_handoff_main.md`
