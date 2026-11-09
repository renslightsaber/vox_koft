# vox 지시문 — 11/09: 워크샵 발표용 전/후 음원(VoxCPM2 Base vs NUX2sN s3000)

> 작성: q3tts 세션 · 사용자 요청 11/09 · **14:00 파일럿 지시문이 끝난 뒤** 실행합니다(GPU 2,3).

## 할 일
- **Base(before)**: VoxCPM2 원본(FT 전)
- **학습 모델(after)**: `A7NUX2sN_xref_LoRA_01_r32_lr5e5` **step_3000** `export_lora`(NUX2sN s3000 · 지금의 생성 모델)
- **설정은 `notebooks/lora/A7NUX2sN_xref_LoRA_01_r32_lr5e5_listening.ipynb` 와 같게** 합니다.
  - 입력 "(v1 지시문)문장" + reference = 위 REF(Controllable cloning · prompt 없음 · D 재측정과 같은 모양)
  - cfg 2.0 · timesteps 10 · `normalize=False` + `ko_tts_norm` · 준비 합성 1회
  - Base 도 같은 "(지시문)문장" 입력으로 만듭니다(전/후 입력이 같아야 공정합니다).
- 결과 폴더: `~/workspaces/tmp_listen/vox_koft/workshop/` → 페이지 `http://127.0.0.1:8765/vox_koft/workshop/index.html`

## 공통 — 세 모델(Qwen3-TTS · CosyVoice3 · VoxCPM2)이 똑같이 맞출 것
- **목적**: 11/12 사내 워크샵 발표(사내 + 협력사 1분)에서 "Base(학습 전) vs 우리 학습 모델" 전/후를 들려줍니다. 세 모델을 **같은 REF · 같은 문장 · 같은 seed** 로 맞춥니다.
- **REF(서비스 목소리)**: `~/workspaces/reference_audios/qwen3tts/set2_trim03_rms20/female_ment_natural_24k_trim03_rms20.wav`
  - 전사문이 필요하면 같은 이름 `.json` 의 `normalized_text` 를 씁니다: "모르는 문제가 있으면 그냥 넘어가지 말고 꼭 질문해줘. 틀린 문제를 다시 푸는 게 실력을 키우는 가장 좋은 방법이야."
- **문장 8개**(원문 그대로 · 정규화는 각 프로젝트 노트북과 같게 · 칸 이름은 각 프로젝트의 감정 표 키에 맞춤)

| id | 칸 | 문장 |
|---|---|---|
| N1 | 중립(Neutrality) | 안녕하세요, 하상일 리더님. 혹시 이게 레퍼런스 오디오와 목소리가 동일한가요? |
| N2 | 중립(Neutrality) | 안녕하세요, 리더님. 혹시 이게 레퍼런스 오디오와 목소리가 동일한가요? |
| HA | 기쁨A(Happy-A) | 드디어 합격했어! 진짜 믿기지가 않아. |
| HB | 기쁨B(Happy-B) | 드디어 합격했어! 진짜 믿기지가 않아. |
| SA | 슬픔A(Sad-A) | 코인에 올인했다가 사기당해서 1억을 날렸어. 아직 대출금도 많이 남았는데 ... |
| SB | 슬픔B(Sad-B) | 코인에 올인했다가 사기당해서 1억을 날렸어 아직 대출금도 많이 남았는데 ... |
| AA | 분노A(Angry-A) | 너 자꾸 나 빡치게 할래? |
| AB | 분노B(Angry-B) | 너 자꾸 나 빡치게 할래? |

  - N1 · N2 는 사용자가 귀로 들어 보고 발표에 쓸 것을 정합니다(N1 에는 실명이 있음 · 발표 자료에는 쓰지 않을 수 있음).
- **seed**: 1986 · 1987 · 1988. 전/후 모두 같은 seed 로 만듭니다.
  - 페이지에는 1986 만 보입니다. 1987 · 1988 은 1986 이 깨졌을 때 교체할 예비 파일입니다.
- **파일**: `<결과 폴더>/wav/<id>_<before|after>_s<seed>.wav`(24kHz · 모노) + `manifest.json`
  - manifest 에 적을 것: 모델 · ckpt 경로 · step · 실제로 넣은 입력(정규화 뒤 문장 · 지시문 · REF · 전사문) · 노브 · seed · 길이
- **재현 확인**: **그 프로젝트의 청음 노트북과 같은 엔진 · 같은 설정**으로 만듭니다. 노트북에 같은 조건의 소리가 이미 있으면 같은지 비교하고, 없으면 1개를 두 번 만들어 같은지 확인합니다. 결과를 manifest 에 적습니다.
- **작은 청음 페이지**: `<결과 폴더>/index.html`
  - 맨 위 REF 재생 · 8줄 × (Base · 학습 모델) · seed 1986
  - 줄마다 메모 · "쓸 만함 / 깨짐" 버튼 · CSV · 볼륨 낮게
- **지킬 것**
  - 학습 · 삭제 · 커밋은 하지 않습니다.
  - upstream 폴더(CosyVoice/ · VoxCPM/ · Qwen3-TTS/)는 수정하지 않습니다.
  - bf16 로 돌립니다.
  - 끝나면 채팅으로 짧게 보고합니다: 페이지 주소 · 재현 결과 · 깨진 소리가 있으면 그 id
