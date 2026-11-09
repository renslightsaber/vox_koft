# CLAUDE.md — vox_koft

> Claude Code 가 이 폴더에서 작업할 때 매 세션 읽는 규칙 · 지형 요약. 진행 상황은 `_handoff/` 최신 문서에.
> 자매 프로젝트 `~/workspaces/q3tts_koft`(Qwen3-TTS 한국어 FT)와 **같은 규칙 · 같은 폴더 틀**을 따른다.

## 1. 작업 규칙 (항상)
- 🗣️ 사용자와 **한국어**로 소통 · 문서도 한국어로, 처음 보는 사람도 이해하기 쉽게. 채팅 표는 좁게 · 시각은 **KST**(서버 시계 UTC + 9).
- 🙋 제안 · 질문은 **먼저 묻고 사용자가 결정**.
- 🚫 **git commit / push 는 사용자가 요청할 때만**(의미 단위로 나눠서). 원격 2곳: `origin` = GitHub(`renslightsaber/vox_koft`, https) · `gitlab` = 사내(`renslightsaber/vox-koft`, ssh). 노트북은 `nbstrip` 필터로 **출력 지운 사본만** 커밋. `VoxCPM/` · `ko-tts-text-normalizer/` 는 submodule — [규칙](docs/rules_in_this_proj/docs_handoff.md) §4.
- 🔒 **`VoxCPM/` 은 공식 저장소(clone) — 수정하지 않는다.** 확장은 이 폴더의 새 코드로. `ko-tts-text-normalizer/` 는 필요 시 수정 가능.
- ⬇️ 모델 · 체크포인트 다운로드는 "사용자 / Claude 실행"을 먼저 묻는다(2026-11-05: Claude 실행 승인 → VoxCPM2 받음).
- ⏱️ 장시간 학습 · 대량 생성 · 삭제는 사용자 승인 뒤 · 짧은 smoke 먼저 · **학습은 사용자가 byobu 에서 실행**(Claude 는 smoke · 백업 · 명령) · **작성한 코드는 실제로 돌려 검증**.
- 📏 규칙 정본은 `docs/rules_in_this_proj/` · seed 1986 · bf16 · `~/workspaces/{q3tts_koft,cv3_koft}` 는 읽기만.
- 🎧 청음이 정본 · 청음 페이지는 `~/workspaces/tmp_listen/` · 재현 확인 뒤 공개.

## 2. 지형
```
vox_koft/
├── VoxCPM/                  # 🔒 OpenBMB/VoxCPM clone (커밋 cce58a5 · 2026-10-08) — editable 설치
├── ko-tts-text-normalizer/  # 한국어 정규화(import ko_tts_norm) — VoxCPM 내장 정규화는 중·영만이라 필수
├── requirements_112/        # env voxcpm 설치 스크립트 · 가이드 · verify_install.py · lock
├── tools/                   # download_ckpts.py · pre_train_backup.py · post_train_backup.py(학습 뒤 백업 · curves) · eval_runs.py(D 재측정 일괄) · vspk.py(가상 화자 일관성 · REF 음향) · bestofn.py(여러 번 만들어 고르기 · 문턱 · 노브) · d_eval_eref.py(감정 ref 판 · 청음 페이지) · val_xref_loss.py · cell_drift.py(칸별 목소리 이동) · cfg_test.py(추론 cfg 시험) · emo_voice_plot.py(감정 ↔ 목소리 그림) · group_split.py(542 vs unseen8 표) · real_cells.py(청음 페이지 진짜 녹음 열) · d_eval_tables.py(best 5 · 상위 5 · 성공 기준) · tb_group.py(TB 그룹 · --where) · listen_pages.py(작은 청음 페이지 — 6명 × 9줄 · 4모델 비교 · run 별 · 메모 · CSV) · instr_test.py(분노B 지시문 시험 · F0) · cross_model.py(모델 간 목소리 유사도 · 섞기 모의실험) · angryB_pitch.py(학습 데이터 칸별 높이) · iv3_compare.py(분노B 재학습 vs 서비스 문구) · make_a7nux2snp.py(B칸 고르기 데이터 · 높이 · 갈라짐) · bcell_compare.py(B칸 출력 높이 · 세기) · happyA_check.py(기쁨A 정답 감정 확인) · ref_sim.py(같은 칸 진짜 녹음 대비 · 칸 쌍 · 기쁨A 섞기 · 12줄 페이지) · pilot_bestofn.py(+ _score · 파일럿: 여러 번 만들어 고르기 · 수율 · 페이지) · crackle_probe.py(갈라짐 지표 시험) · workshop_audio.py(워크샵 전/후 음원 · --set service|0156 · 자동 검사)
├── .claude/agents/          # vox-* 서브 에이전트 10종 — docs/guides/agents_creation_guide.md
├── configs/                 # config.yaml(스키마 · 모르는 키 에러) · hardware/ · accelerate/ · emotion/(지시문 표) · experiments/{sft,lora}/
├── voxko/                   # 우리 패키지(pip install -e . --no-deps)
│   ├── common/              #   config · console · ckpt · rng · sampler · audio · trainer_base(q3ko 이식)
│   ├── data/                #   prepare.py(A7N/542 → 16kHz JSONL · ref 풀) · dataset.py("(지시문)문장" + 무감정 ref)
│   ├── sft/ · lora/         #   전체 FT · LoRA 트레이너(공식 모델 · packer import)
│   ├── engine/synth.py      #   합성(원본 · export_ft · export_lora) · ko_tts_norm · 준비 합성
│   ├── eval/                #   d_eval.py(🔴 성공 판정 = D안 재측정 · 7칸 · 목소리 두 잣대 · 감정 · CER · 깨짐 · 청음 페이지 · --plan trial) · metrics.py
│   └── listen.py            #   청음 노트북 도우미(빌더 notebooks/utils/build_listening_nb.py)
├── scripts/                 # run.sh(train sft|lora — hw ↔ accelerate 짝 · 로그 · 실행 기록 · expandable_segments) · tb.sh(TB 6008 그룹 보기) · runs/(여러 run 연속 실행)
├── tests/                   # CPU 단위 테스트(pytest)
├── notebooks/{sft,lora}/ · exp_reports/{sft,lora}/
├── docs/                    # rules_in_this_proj(규칙) · reference · guides · reports · datasets · plan · decisions · explanations · history
├── _verify/                 # verify_install 합성 결과
└── _handoff/                # 인수인계
```
- 규칙 정본: [docs/rules_in_this_proj/](docs/rules_in_this_proj/README.md)(run 이름 · 계열 폴더 · GPU/포트 · 학습 순서 · 노트북 · 평가 · 문서 · 짧은 요청 약속 [request_templates.md](docs/rules_in_this_proj/request_templates.md)).
- 출력(저장소 밖): ckpt `~/workspaces/outputs/vox_koft/ckpt/<run>/step_N/{export_ft|export_lora}` · TB `~/workspaces/outputs/vox_koft/runs/<run>`(포트 **6008**) · 데이터 `~/workspaces/preprocessed/vox_koft/<데이터>/`.
- 체크포인트: `~/workspaces/ourckpts/voxcpm/VoxCPM2`(2B · 4.96GB · 48kHz 출력).

## 3. 환경
- conda env **`voxcpm`** = q3tts env 복제(Python 3.12 · torch 2.8.0+cu128 · flash-attn 2.8.3) + voxcpm 2.0.3 editable · funasr · gradio · torchcodec 0.7 — [requirements_112/README.md](requirements_112/README.md).
- 서버: RTX PRO 6000 Blackwell ×4(sm_120 · 96GB). q3tts_koft 와 GPU 를 함께 쓰므로 `nvidia-smi` 로 빈 GPU 확인.

## 4. VoxCPM2 핵심 사실 (코드 · 실측으로 확인)
- 합성: `VoxCPM.from_pretrained(<경로>, load_denoiser=False, optimize=False, local_files_only=True)` → `generate(text, prompt_wav_path, prompt_text, reference_wav_path, cfg_value=2.0, inference_timesteps=10, normalize=False, seed=…)`.
- 🔴 **한국어는 `normalize=False` + `ko_tts_norm.normalize(t, profile="qwen3tts")`** — 내장 정규화는 숫자를 영어로 읽음.
- 🔴 같은 seed 라도 **모델 로드 뒤 첫 합성만 값이 다름** → 재현 비교 전 "준비 합성" 1회.
- 모드: Ultimate cloning(ref 소리 + 전사문 + reference — 우리 ICL 과 같은 모양) · Controllable cloning(reference + `(지시문)문장` — 감정 · 말투) · Voice Design(지시문만).
- FT: 공식 `VoxCPM/scripts/train_voxcpm_finetune.py` 는 **쓰지 않고** 우리 트레이너(`voxko`, HF accelerate)에 공식 부품(모델 forward · packer · LoRA)을 끼워 씀 — 이유 · 차이 [docs/reference/voxcpm2_training_code_walkthrough.md](docs/reference/voxcpm2_training_code_walkthrough.md) · 조사 [voxcpm2_overview_finetune.md](docs/reference/voxcpm2_overview_finetune.md).
- 🔴 학습 입력 **16kHz**(VAE encoder) · 출력 48kHz · 감정 = `"(영어 지시문)문장"`(표 `configs/emotion/a7n_vox_v1.yaml` · v2 = 감정마다 4문구, 학습 때 붙임) · ref = 같은 화자 무감정(없으면 ref 없이) · `data.emotion.drop_only` 로 지시문을 뺄 감정을 고름.
- 실행: `./scripts/run.sh train <sft|lora> --exp <yaml> --hw rtxpro6000_<N>gpu [--gpu N | --accel 2gpu_a|b] [--smoke | --resume]` — 안내 [guides/sft_training.md](docs/guides/sft_training.md) · [lora_training.md](docs/guides/lora_training.md).
- 인프라 불변식(q3tts_koft 와 같음): 설정 스키마 · 정책 판단은 모델 빌드 전 · smoke 는 resume 안 함 · TrainState 로 데이터 위치까지 재개 · 스케줄 변경은 플래그 필요 · loss = 합계/전 GPU target 칸 수 · val 은 GPU 수와 무관한 같은 표본 · export → verify → `_COMPLETE`.
- 🔴 **성공 판정 = D안 재측정**(처음 보는 화자 16명 무감정 reference + `(지시문)문장` · 원본과 같은 입력 비교)
  - 원본 기준선(감정 6칸): 목표 1위 67% · e2v 감정 47% · CER 5.7% — 중립 칸을 더한 7칸 "전체"는 71% · 48% · 4.9%(중립이 쉬워서 좋아 보임)
  - 정본 [docs/plan/20261105_success_criteria_and_next_runs.md](docs/plan/20261105_success_criteria_and_next_runs.md)
- 학습 형식 = reference 만(prompt 이어 말하기 없음) · ref 비율(1·2번 0.7 · 3·4번 0.5)의 나머지 = ref 없음.
- 현재(11/09 오후): **B칸 판 sNp 채택 안 함**(사용자 청음 · 음 높이가 세기를 싣고 있었음) → 생성 모델 **NUX2sN s3000 유지** · **파일럿(여러 번 만들어 고르기 · 학습 없음)**: 542 처음 보는 화자 20명 × 70문장 × seed 4 → 문항 82% 남음(기쁨B · 분노B 약 58%) · 고르면 같은 사람 ↑ · 감정 조금 옅어짐 · 기쁨A 는 밝은 추론 문구 (나)가 절충 · 갈라짐 지표 aper(AUC 0.77) · 청음 규칙 2개(학습 끝 C 페이지 · B 페이지 열 추가) · 워크샵 전/후 음원(Base vs NUX2sN s3000 · 서비스 REF 판 + 처음 보는 남성 aihub_0156 판 · 서비스 REF 는 학습 모델이 못 따라감) → **청음 대기**(pilot 20줄 · workshop · workshop_0156 8줄씩) · 11/09 17:45 커밋 · 두 곳 푸쉬 · [파일럿](docs/reports/20261109_pilot_bestofn.md) · 그 전(11/09 낮): **B칸 고르기 판 A7NUX2sNp**(분노B · 기쁨B 의 +8반음 초과 · 갈라진 정답 6,981개 뺌) 학습 · D 재측정 끝 — 분노B +8.2 → +7.0 · 기쁨B +5.7 → +4.8반음 · 세기 그대로 · **s4000 통합 146.2 전체 1위** · 모델은 데이터보다 약 +3반음 · **기쁨A 조사**: 정답은 기쁘게 들림(무감정처럼 3.5%) → 2판 안 함 · 약한 쪽은 모델 · **REF 유사도**: 같은 칸 진짜 녹음 대비 사람 천장의 약 70% · 분노A 쌍이 사람보다 멂 · 기쁨A 섞기 후보 X2Rs6 s2500 · §0 정리 · 커밋 7개 푸쉬(17b9b9b) · 그 뒤 작업은 커밋 안 함 → **청음 대기**(run 별 NUX2sNplr5e5 · ref_check 12줄) · [B칸 판](docs/reports/20261109_A7NUX2sNp_results.md) · [기쁨A](docs/reports/20261109_happyA_data_check.md) · [REF 유사도](docs/reports/20261109_ref_similarity_by_cell.md) · 그 전(11/09 02:20): **밤사이 작업**(vox 세션이 학습까지 직접) — ① 모델 간 유사도: 칸마다 다른 모델로 섞어도 목소리는 거의 그대로(같은 문장 · 다른 모델 0.83 vs 같은 모델 · 다른 문장 0.65 → 모델 차이가 더 작음 · A7NX3 는 빼기) · ② 분노B 정답 녹음은 무감정 +5.4반음 · 모델 +7.9 · ③ **iv3**(학습 문구에서 shouting 뺌): 분노B 안 내려감 → 재학습 불필요 · 서비스 문구(C2 +3.9반음) · ④ **A7NUX2sNn**: 542 A칸 39.6% → 단서는 542 화자 자체 · TB 미리듣기 REF 고침(새 run 부터) · [유사도](docs/reports/20261109_cross_model_similarity.md) · [iv3](docs/reports/20261109_A7NUX2sN_iv3_results.md) · [sNn](docs/reports/20261109_A7NUX2sNn_results.md) · 그 전(11/08 15:15): **작은 청음 페이지 3개**(학습 없음) — A 분노B 지시문 시험(분노A 문구로 −2.8반음 · `!` 는 원인 아님 · 통과 9/24) · B 4모델 비교 · C run 별 13판(6명 × 9줄 · 메모 · CSV) · 재현 모두 통과 → **청음 대기** · [보고서](docs/reports/20261108_listen_pages_and_instr_test.md) · 그 전(11/07 23:25): **A7NUX2sN**(542 0 · 무감정 비율 10.69% 로 되돌림) 학습 · D 재측정 끝 — 542 화자 A칸 감정 41%(A7NUX2s 42% · A7NUX2Rs6 32%) → **원인은 거의 542 행**(무감정 비율 아님) · unseen8 유지 · 점수 1위 수준 s3000 → **청음 판정 대기** · 재개용 파일 정리 끝(11/08 10:09) · [보고서](docs/reports/20261107_A7NUX2sN_results.md) · 최신 `_handoff/` · 그 전(11/07 21:36): **A7NU 두 판**(A7NUX2Rs6 = X2Rs6 + 새 5목소리 · A7NUX2s = 542 0) 학습 · D 재측정 끝 — 새 5목소리 효과 없음 · **542 를 빼면 542 화자 A칸 감정이 돌아옴**(32 → 42% · 목소리 조금 ↓ · 무감정 비율도 함께 ↓ 라 원인 미분리) → **청음 판정 대기** · [보고서](docs/reports/20261107_A7NU_results.md) · 최신 `_handoff/` · 그 전(18:50): **A7NX2Rs · A7NX2Rs6**(542 5.8% · 3.0% · lr 5e-5) 학습 · D 재측정 끝 — 542 화자 A칸 감정은 돌아오지 않음(25~38% · LoRA01 42%) · unseen8 이득 유지 · 전체 점수 1위 X2Rs6 s4500 → **청음 판정 대기** · [보고서](docs/reports/20261107_A7NX2Rs_results.md) · D 재측정 청음 페이지에 unseen8 **진짜 사람 녹음** 열 · 최신 `_handoff/` · 그 전(12:15): A7NX2R 재개용 파일 정리 · **CFG 시험(감정 효과 없음)** · 감정 ↔ 목소리 그림 · 감정 ref 판 청음 페이지 — [보고서](docs/reports/20261107_cfg_test_emo_voice.md) · 그 전(03:50): **다른 칸 ref** A7NX · A7NX3 · **A7NX2R 두 판(lr 1e-4 · 5e-5)** 학습 · 평가 끝 — A7NX2R 은 목소리(B칸 이동 · 1위) 최고 · 감정 약해짐 → **청음 판정 대기** · [A7NX·A7NX3](docs/reports/20261107_A7NX_A7NX3_results.md) · [A7NX2R](docs/reports/20261107_A7NX2R_results.md) · 그 전(15:25): 본 학습 **4개 완료** · D 재측정 18개 모델 + **중립 칸 추가(7칸)** · **가상 화자 일관성**(`tools/vspk.py` — 세트 안 일관성 FT·LoRA 96~97% ≈ 진짜 녹음 98%) · 중립 시범 생성 640개 · **best-of-N**(`tools/bestofn.py` — 5개 중 생성 중립에 가까운 것 고르기 → 센 감정 칸 유사도 0.66 → 0.70 · prompt 모드는 지시문을 읽음) · **청음 판정 대기** — [결과](docs/reports/20261106_A7N_4runs_results.md) · [중립 · 가상 화자](docs/reports/20261106_neutral_cell_virtual_speaker.md) · 최신 `_handoff/`.
- LoRA 는 **V2 LoRAConfig**(공식 추론 스크립트는 V1 import) · sm_120 에서 nano-vllm LoRA 추론 금지(이슈 #326).

## 5. 서브 에이전트 (`.claude/agents/vox-*.md` · 안내 [docs/guides/agents_creation_guide.md](docs/guides/agents_creation_guide.md))
| 에이전트 | 담당 | model |
|---|---|---|
| `vox-sft` | 전체 FT 트레이너 · accelerate · resume | opus |
| `vox-lora` | LoRA 설정 · 저장 · 불러오기 | opus |
| `vox-data-prep` | A7N/542 → 16kHz JSONL · 지시문 표 · ref 풀 | sonnet |
| `vox-metadata-validator` | 데이터 검사 리포트(수정 안 함) | haiku |
| `vox-engine` | 합성(원본 · FT · LoRA) | sonnet |
| `vox-evaluator` | CER · 화자 유사도 · 끝 잘림 · 청음 | sonnet |
| `vox-ser` | 감정 채점(별도 env) | sonnet |
| `vox-debugger` | 오류 진단 | sonnet |
| `vox-env-setup` | env · ckpt | sonnet |
| `vox-progress-logger` | 보고서 · 인수인계 | sonnet |
