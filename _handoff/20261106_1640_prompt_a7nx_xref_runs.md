# vox_koft — A7NX · A7NX3 "다른 칸 ref" LoRA 학습 2개(1차 && 2차): 코드 소폭 수정 · smoke · 학습 전 백업 · byobu 명령

> 📅 2026-11-06 16:40 KST · q3tts_koft 세션에서 작성한 지시문(사용자가 이 폴더의 Claude Code 에 붙여 넣음)
> ⏰ **사용자는 17:00~17:05 퇴근**입니다. 그 전에 **byobu 명령**이 나오면 사용자가 바로 돌립니다. 늦으면 사용자가 밖이나 집에서 돌립니다. **빠르고 정확하게** 해 주세요(명령을 먼저 · 문서는 뒤에).
> 먼저 읽기: `docs/reports/20261106_cell_voice_drift_analysis.md` · `docs/reports/20261106_bestofn_voice_filter.md` · 메타데이터 `~/workspaces/preprocessed/vox_koft/{A7NX,A7NX3}/dataset_info.json` · 만든 도구 `tools/make_a7nx.py`

## 배경(사용자 아이디어 · 결정 11/06)
- 지금 A7N 학습은 "**무감정 ref → 감정 정답**" 한 방향뿐입니다.
  - 무감정이 없는 화자 26명은 ref 없이 학습합니다. 거기에 일부러 ref 를 빼는 30% 도 있습니다.
  - 그래서 "지시문이 목소리도 정한다"를 배워, 센 감정 칸(분노A · 기쁨B · 분노B)에서 목소리가 학습 화자 쪽으로 끌린다는 가설입니다.
  - FT01 과 FT02 가 같은 목소리(0.83)로 수렴한 것도 같은 방향의 근거입니다.
- 아이디어: **같은 사람의 다른 칸 녹음을 ref 로**(예: 기쁨A ref → 분노B 정답 · 슬픔B ref → 중립 정답), **지시문은 정답 칸 것**으로 학습합니다. → "목소리는 ref, 감정은 지시문"(떼어 놓기)
  - 기대 1: 목소리 유지
  - 기대 2: A 만 · B 만 녹음한 화자의 빈칸을 채워 D안 데이터를 만들 수 있음
- 메타데이터(q3tts 세션이 만들고 검증함)

| 데이터 | 행 | 정답 | 화자 | ref |
|---|---|---|---|---|
| `A7NX` (1차) | 104,664 | 104,664 | 96 | 정답마다 1개 고정 |
| `A7NX3` (2차) | 313,992 | 104,664 | 96 | 정답마다 서로 다른 3개 |

- 행마다 `ref_utt · ref_audio(16kHz) · ref_emotion · ref_kind(N/A/B) · ref_duration` 가 들어 있습니다.
- ref 규칙
  - 같은 화자 · **정답과 다른 칸** · 2~10초 · 항상 ref
  - 묶음 비율은 무감정 0.4 · A 0.4 · B 0.2 를 목표로 했고, 없는 묶음을 빼고 다시 나눈 결과 실제는 **N 36.6% · A 37.7% · B 25.7%** 입니다.
  - 무감정 후보 = 학습 Neutrality + 466 ref 풀
  - 칸이 1개뿐인 화자 16명(1,274행)은 뺐습니다.
- 검증: 같은 칸 ref 0 · 자기 자신 0 · 다른 화자 0 · 표본 파일 존재 OK
- 1차가 "고정 ref 1개"인 이유: LoRA_01 은 3,000 step ≈ 0.9 epoch(배치 32)라 정답을 한 번 정도만 봅니다. 그래서 매 step 무작위로 뽑는 것과 사실상 같습니다. 덕분에 1차와 2차가 같은 코드 경로("행에 고정 ref")를 씁니다.
- `val_xref.jsonl`(검증 192행 · 감정 ref 판)도 같은 폴더에 있습니다.

## 1. 코드(최소 수정 · 기존 run 은 그대로 재현돼야 함)
- `voxko/data/dataset.py` `VoxDataset`: 행에 `ref_audio` 가 있으면 **그 ref 를 항상 씁니다**(ref_ratio 무시 · 후보 풀 무시).
  - 길이 계산(`ref_len`)도 그 ref 길이로 합니다.
  - 행에 없으면 지금과 똑같이 동작합니다(기존 설정 재현 확인).
  - 같은 정답이 여러 행(A7NX3)이어도 `utt` 가 같습니다. 난수 키 · 로그에는 `utt_x`(있으면)를 쓰세요.
- 검증 세트는 **기존 `A7N/val.jsonl` + 기존 무감정 풀**을 그대로 씁니다(LoRA_01 과 val loss 를 비교하려고). 가능하면 `val_xref.jsonl` 을 추가 검증 세트로 넣어 주세요(감정 ref 일 때의 loss). 어렵거나 오래 걸리면 빼고 기록만 남깁니다.
- CPU 테스트에 "고정 ref 행"을 하나 추가하고, 기존 테스트가 모두 통과하는지 확인합니다.

## 2. 설정 2개(`configs/experiments/lora/`)
- 공통: `A7N_ref_LoRA_01_r32.yaml` 복제 · lr 1e-4 · r32 · 배치 8 × accum 2 × GPU 2 · 지시문 v1 표 · instruct 비율 그대로
- `A7NX_xref_LoRA_01_r32.yaml`: `train_jsonl: A7NX/train.jsonl` · **max_steps 3000**(LoRA_01 과 같은 노출) · save/eval 250 · keep_last 12
- `A7NX3_xref_LoRA_01_r32.yaml`: `train_jsonl: A7NX3/train.jsonl` · **max_steps 9000**(3배 노출 ≈ A7NX3 0.9 epoch) · save 750 · eval 750 · keep_last 12
- 이름 규칙: `docs/rules_in_this_proj/run_naming.md` 에 데이터 `A7NX · A7NX3` 와 방식 `xref`(다른 칸 ref)를 추가하고, `tools/tb_group.py --where` 로 TB 그룹을 확인합니다.

## 3. 검증 → 백업 → 명령(이 순서 · 빨리)
1. 1GPU smoke 각 1개(`--smoke`): 고정 ref 가 실제로 들어가는지 로그로 확인(ref 길이 · ref_kind 분포 몇 개)
2. **2GPU 최장 배치 메모리 확인**(LoRA_01 은 약 56GB였음 · 이번엔 ref 가 항상 붙어 길어질 수 있음). 넘치면 batch 6 × accum 3 처럼 유효 배치 32 를 유지하며 조정하고 이유를 기록합니다.
3. 학습 전 백업(`tools/pre_train_backup.py`)
4. **byobu 명령**을 채팅 맨 위에 보여 주세요(GPU 2,3 · `--accel 2gpu_b` · 1차 `&&` 2차 · 로그 경로 · 예상 시간).
   - 예상: 1차 약 35분 + 2차 약 1시간 45분(추정)
   - GPU 0,1 은 q3tts SREF3 평가가 약 18시까지 씁니다. 쓰지 마세요.
5. 그 뒤 문서: 데이터 문서(`docs/datasets/` 에 A7NX · A7NX3) · 계획 · 인수인계(§2)

## 4. 학습 끝나면(§1 · 사용자가 내일 아침 요청할 수 있음)
- 종료 확인 · 백업
- D 재측정 두 가지
  - 기존 무감정 ref(LoRA01 과 같은 입력 · 비교 기준)
  - 🆕 **감정 ref 판**: 같은 16명의 감정 녹음을 ref 로 함. 출력이 지시문 감정을 따르는가 · ref 감정이 새어 나오는가 · 목소리
- 가상 화자 일관성(`tools/vspk.py`) · **칸별 "생성 중립 대비" 목소리 이동**(분석 보고서 §3 표와 같은 방식). 핵심 비교 = LoRA01_s1500 vs A7NX vs A7NX3
- 청음 노트북 2개(best 5) · 보고서

## 지켜 주세요
- 한국어 · KST · **학습 실행은 사용자가 byobu 에서**(명령만 주기) · git commit/push 는 사용자 요청 때만 · 삭제 금지 · `VoxCPM/` 수정 금지
- 메타데이터를 다시 만들어야 하면 `tools/make_a7nx.py` 를 고치되 이유를 기록합니다(seed 1986 · 결정적).
