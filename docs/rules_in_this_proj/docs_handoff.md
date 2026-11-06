# 문서 · 인수인계 · git 규칙

> 📅 2026-11-05 · 처음 보는 사람용 · q3tts_koft 규칙을 옮김
> 모음 첫 장: [README](README.md) · 문서 색인: [docs/README.md](../README.md)

---

## 0. 한눈에
- 문서는 **한국어 · 처음 보는 사람용**(짧은 문장 · 표 · "한눈에" 절 · 링크).
- 진행 상황은 CLAUDE.md 가 아니라 **`_handoff/` 최신 문서**에 씁니다. 새 세션은 가장 최근 handoff 부터 읽습니다.
- 🚫 **git commit · push 는 사용자가 요청할 때만**, 원격 두 곳(`origin` GitHub · `gitlab` 사내) 모두에(2026-11-06 저장소 시작).

## 1. 문서를 어디에
| 폴더 | 무엇 | 파일 이름 |
|---|---|---|
| `docs/reports/` | 실험 · 검증 결과 | `YYYYMMDD_<주제>.md` · 그림 `reports/assets/` |
| `docs/plan/` | 계획 · 제안 | `YYYYMMDD_<주제>.md` |
| `docs/decisions/` | 결정과 이유 | `YYYYMMDD_<주제>.md` |
| `docs/explanations/` | 개념 · 생각 정리 | |
| `docs/datasets/` | 데이터 정본 | `<데이터>_vox_koft.md` |
| `docs/history/` | 날짜별 요약 기록 | `YYYYMMDD_<주제>.md` |
| `docs/guides/` | 실행 안내 | |
| `docs/reference/` | 조사 · 외부 자료 · 공식 코드 해설 | |
| `docs/rules_in_this_proj/` | **이 프로젝트의 규칙** | 이 폴더 |

- 새 문서를 쓰면 **색인 `docs/README.md` 를 같이** 고치고, 쓴 뒤 상대 링크가 모두 있는지 확인합니다.

## 2. 쓰는 법
- 머리: `# 제목` → `> 📅 날짜 · 처음 보는 사람용 · 관련 링크` → `---` → `## 0. 한눈에`.
- 시각은 **KST**(서버는 UTC + 9).
- 숫자는 출처(파일 · 로그 · TB)를 남기고, 확인 못 한 것은 "가설 · 확인 안 함"이라고 씁니다.

## 3. 인수인계 (`_handoff/`)
- 이름: `_handoff/YYYYMMDD_HHMM_handoff_main.md`(+ 필요하면 `_handoff_tech_ref.md`) — main = 지금 상태 · 결정 · 실행 명령 · 다음 할 일 · 함정 · 바뀐 파일.
- 목적: `/clear` · `/compact` 뒤에도 바로 이어서 일할 수 있게.

## 4. git (2026-11-06 시작)
- 커밋은 **사용자가 요청할 때만**, **의미 단위**(코드 · 설정+기록 · 노트북 · 문서 · 인수인계)로 나눠서. 메시지는 한국어 + 끝에 `Co-Authored-By` 줄.
- 원격 2곳 · 브랜치 `main`
  - `origin` = GitHub `https://github.com/renslightsaber/vox_koft`(https · 저장된 인증)
  - `gitlab` = 사내 `git@gitlab.tutorusresearch.com:renslightsaber/vox-koft.git`(ssh)
  - `git push origin main` + `git push gitlab main`.
- submodule: `VoxCPM/`(공식 · 커밋 번호만 기록 · 🔒 수정 금지) · `ko-tts-text-normalizer/`(상대 URL `../ko-tts-text-normalizer.git` — 원격마다 자기 저장소).
  - 새로 받기: `git clone --recurse-submodules <url>`
- 노트북은 nbstrip 필터로 **출력 지운 사본만** 올라갑니다(로컬 .ipynb 는 그대로). clone 뒤 필터 등록은 `.gitattributes` 주석 참고.
- 커밋 안 함(.gitignore): 로그 · wav · npz · 체크포인트 · TB(`/runs/`) — 대용량은 원래 저장소 밖(`~/workspaces/outputs/vox_koft` 등).

## 5. 대화
- 진행 안내까지 모두 한국어 · 채팅 표는 **좁게** · 제안 · 질문은 먼저 하고 사용자가 고르게.
