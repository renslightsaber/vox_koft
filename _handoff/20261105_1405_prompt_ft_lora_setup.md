# vox_koft — VoxCPM2 한국어 감정 FT 프로젝트 틀 만들기 (전체 FT + LoRA)

> 📅 2026-11-05 14:05 KST · q3tts_koft 세션에서 작성한 **지시문** — vox_koft 폴더의 Claude Code 에 그대로 붙여 넣어 쓰는 용도입니다.

## 배경
- 이 폴더(`~/workspaces/vox_koft`)는 VoxCPM2(2B · 48kHz · Apache-2.0)를 한국어 감정 TTS로 FT 하기 위한 프로젝트입니다.
- 자매 프로젝트 `~/workspaces/q3tts_koft`(Qwen3-TTS 한국어 감정 FT)와 **같은 구조와 규칙**으로 실험 · 문서 · 인수인계 · 코드를 관리하고 싶습니다.
- 목적: 우리 감정 데이터(A7N 7칸: Neutrality · Happy-A/B · Sad-A/B · Angry-A/B)로 VoxCPM2를 FT 해서
  1. 괄호 지시문 `"(감정 묘사)문장"`으로 감정을 확실히 내게 하고
  2. q3tts_koft 감정 데이터 증강(다른 목소리로 감정 문장 만들기 = D안)에 쓸 수 있는지 확인합니다.
  - 시범에서 VoxCPM2 원본은 지시문만으로는 감정이 약했고, 감정 ref 소리를 쓰면 감정은 나오지만 목소리가 섞이는지 불확실했습니다.
- **전체 FT 와 LoRA 둘 다** 지원해야 합니다.

## 먼저 읽을 것 (순서대로)
1. 이 폴더의 `CLAUDE.md` · `_handoff/` 최신 문서 · `docs/reference/voxcpm2_overview_finetune.md` · `requirements_112/README.md`
2. 자매 프로젝트 구조와 규칙(**읽기만**, 수정 금지)
   - `~/workspaces/q3tts_koft/CLAUDE.md`
   - `~/workspaces/q3tts_koft/docs/rules_in_this_proj/`(run 이름 · 계열 폴더 · GPU/포트 · 학습 순서 · 노트북 · 평가 · 청음 · 문서 · 인수인계 · git)
   - `~/workspaces/q3tts_koft/docs/rules_in_this_proj/request_templates.md`(짧은 요청의 약속)
   - `~/workspaces/q3tts_koft/configs/`(config.yaml 스키마 · hardware/ · accelerate/ · experiments/)
   - `~/workspaces/q3tts_koft/scripts/run.sh`
   - `~/workspaces/q3tts_koft/q3ko/common/`(config · console · ckpt · trainer_base)
   - `~/workspaces/q3tts_koft/.claude/agents/q3-*.md` · `docs/guides/agents_creation_guide.md`
3. VoxCPM 공식 학습 코드(`VoxCPM/` 은 공식 clone — **수정 금지**)
   - `VoxCPM/scripts/train_voxcpm_finetune.py`
   - `VoxCPM/conf/voxcpm_v2/voxcpm_finetune_all.yaml` · `voxcpm_finetune_lora.yaml`
   - `VoxCPM/src/voxcpm/training/` · `VoxCPM/src/voxcpm/model/voxcpm2.py`(LoRAConfig · 괄호 지시문 처리 위치)
   - `VoxCPM/scripts/test_voxcpm_ft_infer.py` · `test_voxcpm_lora_infer.py` · `lora_ft_webui.py`
   - 공식 문서의 FT 안내(README · docs)

## 작업 규칙 (q3tts_koft 와 같음)
- 한국어로 소통하고 문서도 한국어로 씁니다. 처음 보는 사람도 읽기 쉽게, 채팅 표는 좁게, 시각은 KST(서버 UTC + 9).
- 제안 · 질문이 있으면 **먼저 묻고 사용자가 결정**합니다.
- git commit / push 는 요청할 때만 합니다. 아직 git 저장소가 아니며 `git init` 도 사용자 요청 시에만 합니다.
- `VoxCPM/` 은 수정하지 않습니다. 확장은 우리 패키지 쪽 새 코드로 합니다(공식 학습 코드는 import 하거나 우리 쪽에 감싸서 사용).
- 장시간 학습 · 대량 생성 · 삭제는 승인 뒤에 합니다. 짧은 smoke 를 먼저 합니다. **본 학습은 사용자가 byobu 에서 실행**합니다(명령어를 꼭 알려 주기).
- 모델 · 체크포인트를 받을 때는 "사용자가 실행 / Claude 가 실행"을 먼저 묻습니다.
- seed 1986 · bf16 · 청음이 정본입니다.
- 한국어 입력은 `normalize=False` + `ko_tts_norm`(profile `qwen3tts`). 같은 seed 라도 모델 로드 뒤 첫 합성은 값이 달라서 준비 합성 1회 뒤 사용합니다.
- `~/workspaces/q3tts_koft` · `~/workspaces/cv3_koft` 는 읽기만 합니다.

## 할 일 (순서대로 · 단계마다 사용자 확인)
### 1) 서브 에이전트부터
- 위 코드를 읽은 뒤 `.claude/agents/vox-*.md` 를 q3-* 와 같은 형식으로 만듭니다.
  - 예: vox-sft(전체 FT) · vox-lora · vox-data-prep · vox-metadata-validator · vox-engine · vox-evaluator · vox-ser · vox-debugger · vox-env-setup · vox-progress-logger
- `docs/guides/agents_creation_guide.md` 도 만듭니다.
- 목록과 담당은 만들기 전에 사용자에게 보여 주고 확인받습니다.

### 2) 디렉토리 · 규칙 틀
```
vox_koft/
├── VoxCPM/                 # 🔒 공식 clone
├── ko-tts-text-normalizer/
├── configs/                # config.yaml(스키마 · 모르는 키 에러) · hardware/ · accelerate/ · experiments/{sft,lora}/
├── voxko/                  # 우리 파이썬 패키지(pip install -e . --no-deps)
│   ├── common/             # config · console(colorama) · ckpt(저장 · resume 정책) · rng
│   ├── data/               # A7N 등 → VoxCPM JSONL · 감정 지시문 대응표 적용 · 검증
│   ├── sft/                # 전체 FT
│   ├── lora/               # LoRA FT
│   ├── engine/             # 합성(기본 · FT · LoRA 불러오기)
│   └── eval/               # CER · 화자 유사도 · 감정(SER 은 별도 env) · 끝 잘림
├── scripts/run.sh          # train sft | train lora — hw ↔ accelerate 짝 · 로그 · 실행 기록
├── docs/{rules_in_this_proj,guides,reference,reports,plan,decisions,datasets,explanations,history}/
├── exp_reports/<sft|lora>/<ID>/
├── notebooks/              # 청음 노트북(커널 "Python (voxcpm)")
├── _handoff/
└── tests/
```
- 규칙 문서(`docs/rules_in_this_proj/`)는 q3tts_koft 판을 이 프로젝트에 맞게 옮깁니다.
- 구조를 확정하기 전에 사용자에게 보여 줍니다.

### 3) 학습 코드 — 전체 FT 와 LoRA 둘 다
- 공식 학습 코드의 흐름(데이터 · loss · 저장 형식 · LoRA 저장 · 불러오기)을 먼저 문서로 정리합니다(`docs/reference/voxcpm2_training_code_walkthrough.md`).
- 공식 코드에 버그나 우리 환경과 맞지 않는 점이 있으면 표로 정리하고 우리 쪽에서 고칩니다.
- q3tts_koft 학습 인프라와 같은 기능을 갖춥니다.
  - accelerate(1/2/4 GPU) · TensorBoard · colorama 콘솔 · 설정 3축 병합 + CLI `--set`
  - `save_every` · 재개용 파일(`train.resume: {enable, every, keep}`) · `--resume` 때 데이터 위치까지 복원 · 스케줄 변경 가드
  - val(같은 표본) · smoke 모드 · 실행 설정 스냅샷 · export 검증(`_COMPLETE`)
- 전체 FT 는 GPU 당 약 40GB, LoRA 공식 예는 r32 · lr 1e-4 입니다. 서버는 RTX PRO 6000 Blackwell(sm_120 · 96GB) ×4 이며 `nvidia-smi` 로 확인합니다.
- ⚠️ 공식 이슈 #326: sm_120 에서 LoRA + nano-vllm 추론 시 소리가 나빠짐 → 학습 검증은 순수 PyTorch 추론으로 합니다.

### 4) 데이터
- 원천: q3tts_koft 의 A7N
  - `~/workspaces/preprocessed/q3tts_koft/A7/preprocessed_outputs/train_cleanend.codes.jsonl`(106,150문장 · 111 화자 번호 · 124h · 7칸 · 필드 `audio` · `text` · `emotion` · `speaker` · `gender` …)
  - 무감정 보조는 AIHub 542 `~/workspaces/preprocessed/q3tts_koft/542/`
  - 문서: `~/workspaces/q3tts_koft/docs/datasets/README.md`
- 🔴 처음 보는 화자 8명(`q3_extra/val_unseen8`)은 학습에서 제외합니다(평가용).
- VoxCPM2 JSONL(`audio` · `text` · 선택 `ref_audio`)로 변환합니다.
  - 감정은 `"(영어 감정 묘사)문장"` 괄호 지시문으로 넣습니다. 대응표는 `configs/emotion/` 에 둡니다(q3tts `configs/emotion/m1v8_smoke4.yaml` 참고).
  - Neutrality 는 지시문 없이 둘지 사용자에게 묻습니다.
- 오디오 샘플레이트(VoxCPM2 학습 입력 요구)를 공식 코드에서 확인하고 필요하면 변환합니다.
- 데이터 위치(예: `~/workspaces/preprocessed/vox_koft/`)는 사용자에게 확인합니다.

### 5) 검증
- env `voxcpm` 은 학습 모듈 import 까지만 확인했습니다. 실제 학습은 아직 안 돌려 봤으므로 이것을 첫 확인으로 합니다.
- 전체 FT · LoRA 각각
  - 1 GPU smoke
  - resume 테스트(끊고 이어서 loss 가 같은지)
  - 2 GPU 짧은 실행
  - 저장물로 합성 1개
- 결과는 `requirements_112/README.md` 와 보고서에 기록합니다.

### 6) 문서 · 인수인계
- `CLAUDE.md` 갱신 · `docs/README.md` 지도 · `docs/guides/{sft_training,lora_training}.md`
- `_handoff/YYYYMMDD_HHMM_<주제>.md`(KST)
- 본 학습 실행 명령(byobu 용)을 사용자에게 알려 줍니다.
