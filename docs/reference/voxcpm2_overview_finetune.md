# VoxCPM2 조사 — 우리처럼 학습할 수 있나 · 어떤 ckpt · 운영(B) 평가 · 코드스위치에 쓸 수 있나

> 📅 2026-11-05 · 처음 보는 사람용 · 근거 = 공식 저장소 코드(`~/workspaces/vox_koft/VoxCPM`, 커밋 `cce58a5` 2026-10-08) · HF 모델 카드 · GitHub 이슈 · 기술 보고서 · **우리 서버에서 직접 설치 · 합성** · 고객사 검수 run117
> 앞선 조사(라이선스 · 서빙 · 속도): `~/workspaces/q3tts_koft/docs/reference/voxcpm2_finetune_license.md`
> 청음: VoxCPM2 vs 우리 기준 모델 코드스위치 비교 `http://127.0.0.1:8765/q3tts/voxcpm2_cs/index.html`

---

## 0. 한눈에 — 질문별 답

| 질문 | 답 |
|---|---|
| (1) 우리처럼(ICL + 감정 지시문) 또는 cv3 instruct2 처럼 학습 가능? | ✅ **가능** — 공식 **전체 FT · LoRA** 스크립트. 학습 데이터에 `ref_audio` 를 넣으면 "ref 목소리 + 목표 문장" 학습, 문장 앞 **괄호 지시문 `(기쁜 말투로)`** 이 공식 스타일 지시 방식 → cv3 instruct2(지시문 + ref 소리)와 같은 모양 |
| (2) 어떤 ckpt? | **`openbmb/VoxCPM2`**(2B · 48kHz · 30개 언어 · 한국어 포함) — 한국어는 이것만 공식 지원. 1.5 · 0.5B 는 중 · 영만 |
| (3) 운영(B) 만족도 · 불만 | 358건 중 **B 정상 292 · 발음 오류 2 · 기타 1** · 비교 **B 우세 154 / A 우세 4** — 불만은 **숫자 "3개" 를 "삼 개"로 읽음(2건)** · 끝에 불필요한 소리(1건) · 중간 한숨(1건)뿐 |
| (3) 코드스위치 문장 B 는? | ✅ **문제 기록 0건** — "Is this your pencil?" 8번 모두 영어로 바르게(ASR 확인) · "밀크T" 든 34건 모두 "밀크티"(B 지적 2건은 숫자 "3개" 읽기) — 같은 문장에서 A 는 철자 읽기 · 잘림 다수 |
| 라이선스 | Apache-2.0(코드 · 가중치) · 상업 사용 가능 |
| 우리 서버 | ✅ env `voxcpm` 설치 · 합성 확인(VRAM 5.9GB · RTF 0.46 순수 PyTorch) — [설치 가이드](../../requirements_112/README.md) |

⚠️ **B = VoxCPM2 는 추정**입니다(검수 자료에는 "B = 운영"만 있음 · 파일은 24kHz 로 내려와 모델을 파일로 확인할 수 없음). 운영 설정(ref · 정규화 · 옵션)도 모릅니다.

## 1. 학습할 수 있나 — 우리 방식 · cv3 방식과 나란히

### 1-1. 공식 FT 기능
| 항목 | 내용 |
|---|---|
| 실행 | `python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_all.yaml`(전체) · `..._lora.yaml`(LoRA) · 웹 UI `lora_ft_webui.py` |
| 데이터 | JSONL — 필수 `audio` · `text` / 선택 `ref_audio` · `duration` · `dataset_id` (`src/voxcpm/training/data.py`) |
| ref 쓰는 학습 | `ref_audio` 가 있으면 입력 = `[ref 시작, ref 소리, ref 끝, 글자, 소리 시작, 목표 소리, 끝]` · **loss 는 목표 소리에만**(`training/packers.py:322-339`) — 우리 ICL 학습과 같은 생각 |
| 전처리 | 따로 없음 — 학습 중 AudioVAE 로 바로 인코딩(우리처럼 codes 를 미리 뽑지 않음) |
| 학습하는 부분 | AudioVAE(소리 ↔ 잠재)는 **항상 고정** · 전체 FT = LM(base · residual) + LocEnc + LocDiT + 연결층 · LoRA = LM · DiT 의 q/k/v/o (r 32 · lr 1e-4) |
| 하이퍼(v2 기본) | batch 2 × 누적 8 · lr 1e-5(전체) · `max_batch_tokens 8192` · loss = diffusion + stop |
| 필요 VRAM | 공식 답변: **전체 FT 는 GPU 1장에 약 40GB**(여러 장으로 나눠지지 않음 · 3090×4 는 OOM → LoRA) — 우리 96GB 는 충분 ([#335](https://github.com/OpenBMB/VoxCPM/issues/335)) |
| 데이터 양 경험 | 약 15분으로 목소리 따라 하기 가능 · 1시간 여러 epoch 는 감정을 배우지만 불안정 → 추론 때 ref 를 넣으면 안정 ([#271](https://github.com/OpenBMB/VoxCPM/issues/271)) |

### 1-2. 우리 · cv3 방식에 대응시키면
| | 우리 Qwen3(q3tts_koft) | cv3_koft(CosyVoice3 instruct2) | VoxCPM2 |
|---|---|---|---|
| 목소리 | ICL: ref 소리 + ref 전사문 | ref 소리(화자 벡터 · 프롬프트 토큰) | **Ultimate cloning**: ref 소리 + 전사문 (+ reference) / **Controllable cloning**: reference 소리만 |
| 감정 | 입력 앞 지시문(학습으로 가르침 · 0.6B 는 원래 미지원) | instruct 문장 | **문장 앞 괄호 지시문** `(cheerful tone)…`(원래 지원) |
| 학습 데이터 | ref + 감정 지시문 + 정답 | instruct + ref + 정답 | `text` = `(지시문)문장` · `ref_audio` = 같은 화자 다른 녹음 · `audio` = 정답 |
| 주의 | — | — | 공식 CLI 는 **지시문과 prompt 전사문을 같이 못 씀**(`cli.py:119-120`) → 감정 지시는 Controllable cloning(전사문 없는 ref) 쪽 |

→ **"ref 목소리 + 감정 지시문" FT 는 데이터 모양만 맞추면 바로 가능**. 우리 A7N(7칸 감정 + ref 규칙)을 `audio/text/ref_audio` JSONL 로 바꾸는 변환기 하나면 됩니다.

### 1-3. 조심할 점
- **한국어 텍스트 정규화가 없음**: 내장 정규화는 중 · 영만(`utils/text_normalize.py`) — 한국어 문장에 `normalize=True` 면 숫자를 영어로 읽음 → **`normalize=False` + 우리 `ko_tts_norm`** 으로 넣어야 함.
- 출력 48kHz · 학습 VAE 입력 16kHz(설정 `sample_rate: 16000` 과 VAE 가 맞지 않으면 assert).
- 매번 결과가 달라 공식도 1~3회 생성 권장 · 길거나 감정이 센 입력에서 가끔 불안정(모델 카드).
- 끝을 못 맺고 소리가 더 붙는 사례(베트남어 [#352](https://github.com/OpenBMB/VoxCPM/issues/352)) · 단어 반복/누락(페르시아어 #394) — 한국어 보고는 못 찾음.
- **LoRA + nano-vllm 을 Blackwell(sm_120)에서 오래 돌리면 음질이 점점 나빠지는 미해결 보고**([#326](https://github.com/OpenBMB/VoxCPM/issues/326)) — 우리 서버도 sm_120.

## 2. 체크포인트
| HF id | 크기 | 출력 | 언어 | FT |
|---|---|---|---|---|
| **`openbmb/VoxCPM2`** ⭐ | 2B(4.96GB) | 48kHz | 30개(한국어 포함 · 언어 태그 불필요) | 전체 · LoRA |
| `openbmb/VoxCPM1.5` | 0.6B | 44.1kHz | 중 · 영 | 전체 · LoRA |
| `openbmb/VoxCPM-0.5B` | 0.5B | 16kHz | 중 · 영 | 전체 · LoRA |
- 한국어 수치(기술 보고서 [arXiv 2606.06928](https://arxiv.org/abs/2606.06928) · 낮을수록 좋음): CV3-eval ko CER 5.69(CosyVoice3-1.5B 와 같음) · MiniMax-MLS ko WER 1.96(Qwen3-TTS 1.7B 1.76 이 조금 나음).
- 학습 데이터 200만 시간+ (대부분 중 · 영 · 나머지 28개 언어는 언어당 약 1천~5만 시간) · **코드스위치 학습 · 평가 언급 없음**.
- 우리 서버: `~/workspaces/ourckpts/voxcpm/VoxCPM2`.

## 3. 운영(B) 평가 — 고객사 검수 run117 (2026-10-16 · A/B 358건)
> 자료 `~/workspaces/run117_검수결과/results/대화+음성_AB비교_전체.csv` · A = 개발 · 스테이지(CosyVoice3 base) · B = 운영(VoxCPM2 추정 · **학습 안 한 상태**)

| | A | B |
|---|---|---|
| 정상 | 201 | **292** |
| 누락 · 잘림 | 83 | **0** |
| 발음 오류 | 36 | 2 |
| 기타 | 6 | 1 |
| 판정 안 함 | 32 | 63(검수자2 — 대부분 "B 우세") |
| 비교 | A 우세 4 | **B 우세 154** · 동등 199 |

**B 에 대한 불만 — 전부(4건)**
1. "총 **3개**의 상장" → "삼 개"로 읽음 ×2 — 숫자 + 단위 읽기(정규화 문제).
2. "B 음성 뒷부분에 불필요한 음성 나옴" ×1 — 끝을 못 맺는 사례(위 #352 와 같은 종류).
3. "B 선생님 피곤하신지 중간에 한숨" ×1 — 숨소리 · 운율.

**코드스위치 문장(영어 · 브랜드)** — 영어가 든 메시지 43건(고유 26문장)
| 문장 | B | A |
|---|---|---|
| "Is this your pencil?" (8건) | 문제 기록 0 · ASR 8/8 "Is this your pencil" | 철자 읽기("아이 에스 디스") · 뒤 잘림 7/8 |
| "밀크T" (34건 · 브랜드) | 브랜드 문제 기록 0 · ASR 34/34 "밀크티"(지적 2건은 같은 문장의 "3개" 읽기) | 첫 문장 뒤 잘림 다수(코드스위치 탓보다 A 의 잘림 문제) |
| 학생 이름 "김하윤" | 문제 기록 0 | 이름 발음 오류 5건 |
- ASR = whisper large-v3-turbo(한국어 모드) — 참고용. 검수자 메모와 같은 방향.
- **결론: 운영 B 는 학습 없이도 고객 문장 수준의 코드스위치(짧은 영어 표현 · 브랜드)를 문제없이 읽었다.** 불만은 숫자 읽기 · 드문 끝맺음 실패.

## 4. 우리 서버에서 직접 — VoxCPM2 vs 우리 기준 모델 (같은 서비스 ref · 같은 입력)
- 페이지: `~/workspaces/tmp_listen/q3tts/voxcpm2_cs/`(`make.py vox` → `make.py ours`) — 15문장(고객 3 · 코드스위치 8 · 긴 문장 1 · 🟧 운영 B 불만 문장 3 + 고객 3번 줄 = 불만 4건, 운영 B 원본 소리 · 검수 메모 함께) × 열 4개(VoxCPM2 무지시 · VoxCPM2 Happy-A · 기준 s2250 무감정 · 기준 s2250 Happy-A) · seed 1986 · 둘 다 재현 확인 ✅(11/05 Happy-A · 불만 줄 추가).
- Happy-A: 우리 = 노트북 지시문 “Speak in a relaxed, hopeful tone, low-key.” · VoxCPM2 = Controllable cloning(reference 소리만 + “(relaxed, hopeful tone, low-key)”) — 지시문은 ref 전사문과 같이 못 써서 무지시 열(Ultimate cloning)과 ref 방식이 다름.
- VoxCPM2 = 사전학습 그대로 · Ultimate cloning · cfg 2.0 · 10 steps / 우리 = `A7N_icl_2STG_S2_emotion_s1000` s2250 · ICL streaming · 노트북 엔진.
- ASR(참고 — whisper 한국어 모드는 영어를 한글로 옮겨 적기도 해서 판정용이 아님):

| # | 문장(요약) | VoxCPM2 | 우리 s2250 |
|---|---|---|---|
| 1 | 고객: "Is this your pencil?"이에요… | ✅ 영어 | ✅ 영어 |
| 2 · 3 | 고객: 밀크T · 3개 | ✅ 밀크티(3개는 ASR 로 구별 못 함) | ✅ 밀크티(〃) |
| 4 | Is this your pencil? 을 배웠어요 | ✅ 영어 | ❌ "SSU 펜슬" |
| 5 | Good job! | ✅ 영어 | 한글 표기(굿잡) |
| 6 | Thank you very much | 한글 표기 | ✅ 영어 |
| 8 | key point · user experience | 한글 표기 | 반 영어 |
| 9 | meeting · Zoom | ✅(줌) | ❌ "숨" |
| 11 | beta version | ❌ "베라불신" | ✅(베터 버전) |
| 12 | 긴 문장 | ✅ 끝까지 | ✅ 끝까지 |
- **ASR 만으로는 VoxCPM2 가 확실히 낫다고 말할 수 없음** → **청음이 정본**. 특히 4 · 9 · 11 번을 들어 주세요.

## 5. "throughput 이슈라 FT 필요 없다"는 말에 대해 (Claude 앱 답변 검토)
- 앱 답변의 요지(공식 수치상 처리량은 나쁘지 않음 · FT 는 품질용이지 처리량과 무관 · LoRA 서빙은 약간 느려짐 · 사내 GPU · 한국어로 직접 재야 함)는 **맞습니다**.
- 덧붙일 것: ① 우리 서버 순수 PyTorch 첫 측정 RTF 0.46(서빙 엔진 없이) ② **LoRA + nano-vllm 을 sm_120(Blackwell)에서 오래 돌리면 음질 저하 미해결 보고**(#326) — FT 를 LoRA 로 서빙한다면 이 점을 먼저 확인 ③ 전체 FT 로 만든 가중치는 LoRA 비용 없이 원래 속도.

## 6. 우리 코드스위치 문제에 쓰는 방법 (제안 — 사용자 결정 필요)
| 안 | 방법 | 장점 | 위험 |
|---|---|---|---|
| A. **코드스위치 "선생님" 데이터** | VoxCPM2 로 코드스위치 · 긴 문장을 우리 학습 화자 ref 로 합성(수천 문장) → Qwen3 FT 에 소량(약 10%) 섞기 | 학습 데이터에 영어 0문장 문제를 바로 메움 · Apache-2.0 | 합성음 목소리가 원 화자와 조금 다를 수 있음 → 목소리 이동 · 합성 티. 공정한 잣대로 검사 필요 |
| B. **PO 의 chosen 후보** | 같은 문장을 우리 모델 seed 8개로 합성 + VoxCPM2 합성 → 맞게 읽은 쪽을 chosen | 우리 목소리 유지(우리 모델 출력이 주) | 채점기(영어 바르게 읽었나) 신뢰도부터 |
| C. **VoxCPM2 를 직접 FT**(cv3 처럼) | A7N 감정 데이터로 전체 FT(40GB) · 감정 지시문 = 괄호 | 코드스위치 · 끝맺음은 원래 강함 · 감정 · 목소리를 더함 | 2B · diffusion — 서빙 비용 · 새 학습 코드 · 시간 |
| D. 문장별 라우팅 | 영어가 든 문장만 VoxCPM2 | 바로 적용 | 목소리가 문장마다 달라짐 — 서비스로는 어려움 |
- 숫자 읽기(3개 → 삼 개)는 어느 쪽이든 **`ko_tts_norm` 으로 미리 "세 개"로 풀어 넣으면** 해결.
