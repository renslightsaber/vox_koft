# TensorBoard 를 그룹(학습 방식 → 데이터)으로 보기

> 📅 2026-11-06 · 처음 보는 사람용 · 코드 `tools/tb_group.py` · 실행 `scripts/tb.sh` · q3tts_koft 의 같은 도구를 옮김

---

## 0. 한눈에
```bash
# 서버(byobu 창 하나) — 포트 6008 · 5분마다 트리 갱신 · 30초마다 새 값 반영
cd ~/workspaces/vox_koft && ./scripts/tb.sh
# 맥북 — 터널을 연 뒤 브라우저에서 http://localhost:6008
ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L 6008:localhost:6008 112
```
- 서버 별칭 `112` 는 q3tts_koft 문서와 같습니다(맥북 `~/.ssh/config` 의 Host 이름).
- `-N` 은 명령 없이 터널만 엽니다. 창을 닫으면 터널도 끊깁니다.
- `ServerAliveInterval=30` 은 오래 켜 둬도 끊기지 않게 합니다. `ExitOnForwardFailure=yes` 는 6008 이 이미 쓰이고 있으면 바로 알려 줍니다.

## 1. 폴더 구조
```
~/workspaces/outputs/vox_koft/
├── runs/                          ← 원본(학습 코드가 씀 · 그대로 둠)
│   └── A7N_ref_FT_01/events.out.tfevents.*
└── runs_grouped/                  ← 링크 모음(tools/tb_group.py 가 만듦 · 통째로 지워도 원본은 안전)
    ├── FT/A7N/A7N_ref_FT_01        → ../../../runs/A7N_ref_FT_01   (폴더 symlink)
    ├── FT/A7N/A7N_ref_FT_02_iv2
    ├── LoRA/A7N/A7N_ref_LoRA_01_r32 · A7N_ref_LoRA_02_r32_iv2
    ├── LoRA/A7N542/…              (542 섞기 판이 생기면)
    ├── smoke/A7N100/…             (이름 끝이 _smoke)
    └── _misc/                     (규칙에 안 맞는 이름)
```
- 규칙: run 이름 `데이터_조건_학습방식_번호_덧붙임` 에서 학습 방식(`FT` · `LoRA`, 끝이 `_smoke` 면 smoke)과 데이터(`A7N` · `A7N100` · `A7N542` …)를 뽑습니다 — [run_naming.md](../rules_in_this_proj/run_naming.md).
- 매핑 표: `~/miniforge3/envs/voxcpm/bin/python tools/tb_group.py --dry-run`

### 왜 폴더 symlink 인가 (2026-11-06 00:10 KST 실측)
| 방식 | TensorBoard 2.21 에서 보임 | resume 으로 새 기록 파일이 생기면 |
|---|---|---|
| **폴더 symlink**(채택) | ✅ | ✅ 자동으로 읽음(점 5 → 9) |
| 진짜 폴더 + 기록 파일마다 symlink | ✅ | ❌ 트리를 다시 만들어야 보임 |
- "TensorBoard 는 폴더 symlink 를 따라가지 않는다"는 전제는 이 버전(env voxcpm · q3tts 모두 2.21.0)에서는 맞지 않습니다.
- q3tts_koft 의 최종 구현도 폴더 symlink 입니다.

## 2. 서버에서 띄우기 (`scripts/tb.sh`)
| 명령 | 뜻 |
|---|---|
| `./scripts/tb.sh` | 그룹 트리 · 포트 6008 · localhost 만 · 5분마다 트리 갱신 |
| `./scripts/tb.sh --port 6011` | 다른 포트 |
| `./scripts/tb.sh --flat` | 원래 `runs/` 그대로(그룹 없음) |
| `./scripts/tb.sh --every 120` | 트리 갱신 주기(초) |
- 내부 옵션은 `--reload_interval 30 --reload_multifile true` 입니다. 학습 중인 run 도 30초마다 새 값이 반영되고, resume 으로 생긴 기록 파일도 함께 읽습니다.
- Ctrl-C 로 끄면 트리 갱신 루프도 같이 끝납니다.
- 포트 6006 · 6007 은 다른 프로젝트(cv3_koft · q3tts_koft)가 씁니다.

## 3. 화면에서 그룹 켜고 끄기
- **왼쪽 run 검색창(Filter runs, 정규식)**
  - `^FT/` → 전체 FT 둘 · `^LoRA/` → LoRA 둘
  - `_iv2` → 웹 조사판(3 · 4번)만 · `_01` → 1 · 2번만
  - `^(FT|LoRA)/A7N/` → 본 학습 4개 · `^smoke/` → smoke
- 검색한 뒤 목록 위 체크박스로 보이는 run 을 한꺼번에 켜고 끌 수 있습니다.
- **Time Series 탭 설정(톱니)**: 버전에 따라 run 색 묶음 기능이 있으면 정규식 `^([^/]+)/` 로 학습 방식별 같은 색을 줄 수 있습니다.
- 볼 곳: `train/loss_diff` · `train/loss_stop` · `val/*` · **`val_unseen/*`(처음 보는 화자 8명)** · `train/ref_frac` · `media/pred_audio_*`(500 step 마다 합성).

### ⚠️ media 소리(미리듣기) — 11/08 이전 run 은 목소리 비교에 쓰지 마세요
- **11/08 이전 run**(A7NUX2sN_xref_LoRA_01 까지 전부)의 `media/pred_audio_*` 는 **REF 없이 만든 소리**입니다.
  - 문장이 `A7N/val.jsonl` 처음 2줄(466 화자 `a466_0001`)이었는데, 그 화자가 무감정 ref 풀에 없어서 REF 가 붙지 않았습니다.
  - VoxCPM2 는 REF 가 없으면 아무 목소리나 만들기 때문에, `pred_audio` 가 `target_audio` 와 **성별까지 다를 수 있습니다**.
  - **학습 자체는 정상입니다**(xref 학습 행은 100% 고정 ref 를 가짐). 미리듣기만 잘못 만들어진 것입니다. 지난 run 의 media 는 다시 만들지 않았습니다.
- **11/09 부터**(A7NUX2sN_xref_LoRA_02_r32_lr5e5_iv3 · A7NUX2sNn_xref_LoRA_01 부터) 미리듣기 3줄은 REF 가 확실히 있는 행입니다.
  - `val_xref` 처음 2행(행에 고정 ref) + `val_unseen` 1행(풀 ref). `val_xref` 가 없는 설정이면 val 에서 풀 ref 가 있는 행만 씁니다.
  - 캡션 끝에 `REF <utt> (<화자> · <칸>) | <출처>` 가 붙고, REF 소리는 `media/ref_audio_{i}` 로 처음 한 번 기록됩니다 → TB 에서 REF 와 나란히 들어 보세요.
  - smoke 때 확인(WavLM 코사인): 세 줄 모두 REF 와 같은 성별이고, pred↔REF 0.53~0.72 로 같은 화자 진짜 녹음끼리(REF↔정답 0.52~0.83)와 같은 수준이었습니다.
  - 실제 학습(A7NUX2sN iv3 · TB 에 기록된 소리) step 4500: pred↔REF 0.65 · 0.66 · 0.75(WavLM). 예전 run(A7NUX2sN_01) 의 media 는 pred↔정답 0.14 · 0.30 으로 다른 사람이었습니다.
  - 코드: `voxko/sft/trainer.py` `media_rows()` · `voxko/common/trainer_base.py` `log_media()` · smoke 는 이제 media 문장 전부를 합성해 `<run>/_smoke/` 에 REF 와 함께 저장합니다.

## 3-1. 새 규칙 (2026-11-06 · q3tts_koft 와 같음)
- `scripts/run.sh` 는 학습을 시작할 때 **그룹 경로를 알려 주고**(`tensorboard: ./scripts/tb.sh → 그룹 runs_grouped/FT/A7N/<run>`), 약 2분 30초 뒤 기록 폴더가 생기면 **그룹 트리에 자동으로 연결**합니다(tb.sh 를 안 띄워도 됨).
  - ⏳ 이 run.sh 는 `scripts/run.sh.next` 로 준비해 두었습니다. **지금 돌고 있는 4개 연속 학습이 끝난 뒤 교체**합니다(돌고 있는 셸이 읽는 파일이라서).
- **새 학습 방식 이름**(예: DPO)은 `tools/tb_group.py` 의 `method_of()` 에 한 줄 추가합니다. 안 넣으면 `_misc/` 로 가고 run.sh 가 경고합니다.
  - 미리 확인: `~/miniforge3/envs/voxcpm/bin/python tools/tb_group.py --where <run>` → 예 `FT/A7N/A7N_ref_FT_01`
- 연결 작업은 출력 통로를 떼어 낸 백그라운드라 run.sh 출력을 파이프로 받아도 기다리지 않습니다.
  - q3tts_koft 판은 출력 통로를 떼지 않아서 파이프로 받으면 150초 동안 기다립니다(그쪽은 읽기만 · 보고만).

## 4. 새 run 이 생기면
- `scripts/tb.sh` 가 켜져 있으면 5분 안에 트리에 들어가고, 30초 안에 화면에 보입니다.
- 바로 보고 싶으면 `~/miniforge3/envs/voxcpm/bin/python tools/tb_group.py` 를 실행합니다.
- 원본 run 을 지우면 다음 갱신 때 깨진 링크와 빈 폴더가 정리됩니다(링크만 지움).
