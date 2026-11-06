# agents_creation_guide.md — vox_koft 서브 에이전트 안내

> 📌 **목적**: vox_koft 의 Claude Code 서브 에이전트 10종(`.claude/agents/vox-*.md`)이 **무엇을 맡는지, 어떻게 쓰고 고치는지** 한눈에 보는 문서입니다.
> 🧭 **정본은 `.claude/agents/vox-*.md` 입니다.** 이 문서는 프롬프트 본문을 복제하지 않습니다(복제하면 한쪽만 고쳐져 어긋나기 쉽습니다).
> 🧩 참고 원형: `~/workspaces/q3tts_koft/.claude/agents/q3-*.md`(10종) — 같은 형식.
> 📅 작성: 2026-11-05

---

## 1. 서브 에이전트란?

Claude Code 가 특정 분야 작업을 **전문 역할을 가진 별도 에이전트**에게 맡기는 기능입니다.
- 각 에이전트는 자기 프롬프트(규칙 · 지식), 쓸 수 있는 도구(tools), 모델(model)을 가집니다.
- 메인 세션은 요청 내용을 보고 `description` 이 맞는 에이전트를 **자동으로** 고릅니다.
- 직접 지정할 수도 있습니다: *"vox-data-prep 에이전트로 A7N 변환해줘"*.

## 2. 에이전트 목록

| 에이전트 | 이런 말을 하면 호출됨 | 담당 | Model |
|---|---|---|---|
| **vox-sft** | 학습, FT, 전체 FT, accelerate, resume | 전체 FT 트레이너 · 인프라 | opus |
| **vox-lora** | LoRA, rank, 어댑터 | LoRA 설정 · 저장 · 불러오기 | opus |
| **vox-data-prep** | 전처리, JSONL, 16kHz, 지시문 표 | A7N/542 → 16kHz JSONL · ref 풀 | sonnet |
| **vox-metadata-validator** | 검증, 데이터 체크 | 데이터 검사 리포트(**수정 안 함**) | haiku |
| **vox-engine** | 추론, 합성, voice clone | 원본 · FT · LoRA 불러오기 · 합성 | sonnet |
| **vox-evaluator** | 평가, CER, SIM, 청음 | 발음 · 화자 · 감정 · 끝 잘림 · 청음 | sonnet |
| **vox-ser** | SER, 감정 채점 | 감정 채점(**별도 env**) | sonnet |
| **vox-debugger** | 에러, OOM, NaN, 이상해요 | 학습/추론 오류 진단 | sonnet |
| **vox-env-setup** | 설치, 환경, conda, 다운로드 | env voxcpm · ckpt | sonnet |
| **vox-progress-logger** | 기록, 리포트, 인수인계 | exp_reports · 보고서 · _handoff | sonnet |

- Tools: validator 는 Edit 없음(리포트만), debugger 는 Write 없음, env-setup 은 Read · Bash · Grep · Write, 나머지는 Read · Write · Edit · Glob · Grep · Bash.
- 모델 배분은 q3 와 같습니다: 설계 난도가 높은 학습(sft · lora)은 opus, 단순 검사는 haiku, 나머지는 sonnet.

## 3. q3 → vox 대응 관계

| q3 (Qwen3-TTS) | vox (VoxCPM2) | 왜 바뀌었나 |
|---|---|---|
| q3-sft | **vox-sft** | 모델 구조가 다름(LM + 연속 latent DiT · CFM loss + stop loss). 공식 학습 부품을 import 해서 우리 트레이너에 끼움 |
| q3-po | **vox-lora** (새 역할) | 이 프로젝트는 아직 선호 최적화 계획이 없고, 대신 **전체 FT 와 LoRA 둘 다** 가 요구사항 |
| q3-data-prep | vox-data-prep | 24kHz + codes → **16kHz wav**(VAE 입력). 감정은 instruct 칸이 아니라 **`(지시문)문장` 괄호 텍스트**, ref 는 무감정 문장 |
| q3-metadata-validator | vox-metadata-validator | 16kHz 검사 · 지시문 표 누락 · ref 풀 · val_unseen8 누출 검사 추가 |
| q3-engine | vox-engine | API 가 `generate(text, reference_wav_path, …)` 하나. FT/LoRA 불러오기 · V1/V2 LoRAConfig 함정 |
| q3-evaluator | vox-evaluator | 출력 48kHz. D안(무감정 ref + 지시문) 지표 추가 |
| q3-ser | vox-ser | q3tts_ser env 를 **쓰기만** 함(전용 env 는 필요 시 결정) |
| q3-debugger | vox-debugger | 16kHz assert · 8192 잘림 · LoRA/nano-vllm(#326) · 숫자 영어 읽기 |
| q3-env-setup | vox-env-setup | env `voxcpm`(q3tts 복제) |
| q3-progress-logger | vox-progress-logger | 계열 `sft` · `lora` |

## 4. 모든 에이전트가 공유하는 사실

1. **작업 규칙**: 한국어 소통 · 결정은 사용자에게 먼저 묻기 · git commit/push 금지 · 다운로드 전 실행 주체 질문 · 장시간 작업 · 대량 생성 · 삭제는 승인 뒤 · **본 학습은 사용자가 byobu 에서 실행** · 작성한 코드는 실제로 돌려 검증.
2. 🔒 **`VoxCPM/` 수정 금지.** 확장은 `voxko/` 새 코드로. q3tts_koft · cv3_koft 는 읽기만.
3. **디렉토리**: 코드 `voxko/{common,data,sft,lora,engine,eval}` · 레시피 `scripts/` · 설정 `configs/` · 기록 `exp_reports/` · 규칙 `docs/rules_in_this_proj/`.
4. **환경**: env `voxcpm`(torch 2.8.0+cu128). SER 은 별도 env, 파일로만 연결.
5. **하드웨어는 확인 후 사용**: RTX PRO 6000 Blackwell ×4(sm_120 · 96GB) · q3tts_koft 와 공유.
6. **체크포인트**: `~/workspaces/ourckpts/voxcpm/VoxCPM2`.
7. **데이터 규약**: 학습 입력 16kHz · 출력 48kHz · 한국어는 `normalize=False` + `ko_tts_norm(profile="qwen3tts")` · 같은 seed 라도 로드 뒤 첫 합성은 다름(준비 합성 1회).
8. **seed 1986 · bf16 · 청음이 정본.**

## 5. 사용 방법

- **자동**: 그냥 요청하면 됩니다. 예) *"학습 JSONL 검증해줘"* → `vox-metadata-validator`.
- **명시**: *"vox-lora 에이전트로 r16 실험 설계안 제안해줘"*.
- **목록 확인/편집**: Claude Code 에서 `/agents`.

## 6. 에이전트를 고치거나 추가할 때

1. `.claude/agents/vox-<이름>.md` 를 직접 편집한다(맨 위 `---` 사이가 frontmatter: `name` · `description` · `tools` · `model`).
2. `description` 은 **"언제 호출되는지"** 가 드러나게 쓴다(트리거 단어 포함).
3. 수시로 바뀌는 진행 상황은 에이전트 파일에 쓰지 않는다 → `_handoff/` 에.
4. 이 문서의 표(§2)와 `CLAUDE.md` 에이전트 표도 함께 갱신한다.
5. 폴더를 만들거나 이름을 바꾸면 각 에이전트의 경로 문구와 `CLAUDE.md` 지형도 함께 갱신한다.
