#!/usr/bin/env python
"""청음 노트북 빌더 — notebooks/<sft|lora>/<run>_listening.ipynb (run 하나에 하나 · 커널 "Python (voxcpm)")

    ~/miniforge3/envs/voxcpm/bin/python notebooks/utils/build_listening_nb.py --runs A7N_ref_FT_01 [...] [--force]

🔒 이미 있는 노트북은 덮어쓰지 않는다(사용자가 실행 · 수정했을 수 있음) — 새로 만들려면 --force(사용자 확인 뒤만).
셀 2(노브)의 best 5 주석 = D 재측정 참고 점수 상위 + TensorBoard 처음 보는 화자 val 상위 + 이상(끝 잘림 등) 비율.
검증: nbformat.validate + LISTEN_SMOKE=1 jupyter nbconvert --execute (표를 1칸 · 2열로 줄여 실행만 확인).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
H = Path.home() / "workspaces"
D_EVAL = H / "tmp_listen/vox_koft/d_eval"
SHORT = {"A7N_ref_FT_01": "FT01", "A7N_ref_LoRA_01_r32": "LoRA01", "A7N_ref_FT_02_iv2": "FT02iv2", "A7N_ref_LoRA_02_r32_iv2": "LoRA02iv2",
         "A7NX_xref_LoRA_01_r32": "A7NX", "A7NX3_xref_LoRA_01_r32": "A7NX3",                       # 2026-11-07 다른 칸 ref
         "A7NX2R_xref_LoRA_01_r32": "X2R01", "A7NX2R_xref_LoRA_02_r32_lr5e5": "X2R02lr5e5",
         "A7NX2Rs_xref_LoRA_01_r32_lr5e5": "X2Rs01lr5e5",
         "A7NX2Rs6_xref_LoRA_01_r32_lr5e5": "X2Rs6lr5e5"}


def best_info(run: str) -> dict:
    """d_eval summary(참고 점수) + run_backup/analysis.json(TB) → best 5 · 주석 줄."""
    sys.path.insert(0, str(ROOT))
    from voxko.listen import steps as all_steps
    track = "lora" if "_LoRA_" in run else "sft"
    lines, pick = [], []
    summ = json.load(open(D_EVAL / "summary.json")) if (D_EVAL / "summary.json").exists() else {"models": {}}
    pre = SHORT.get(run, run) + "_s"
    ev = {int(t[len(pre):]): m for t, m in summ["models"].items() if t.startswith(pre)}
    base = summ["models"].get("base", {}).get("전체", {})
    if ev:
        order = sorted(ev, key=lambda s: -ev[s]["전체"]["score"])
        pick += order[:5]
        lines.append("D 재측정 참고 점수 상위(감정 + 목표1위(공정) − CER − 이상): " + " · ".join(
            f"s{s} {ev[s]['전체']['score']:.0f}(감정 {ev[s]['전체']['emo_hit']:.0f}% · 목표1위 W{ev[s]['전체']['top1']:.0f}% · 이상 {ev[s]['전체'].get('odd', float('nan')):.0f}%)"
            for s in order[:5]))
        if base:
            lines.append(f"원본 기준선: 감정 {base['emo_hit']:.0f}% · 목표1위 W{base['top1']:.0f}% · 이상 {base.get('odd', float('nan')):.0f}% · 점수 {base['score']:.0f}")
    an = ROOT / "exp_reports" / track / run / "run_backup/analysis.json"
    if an.exists():
        b = json.load(open(an)).get("best", {})
        if "val_unseen/loss" in b:
            lines.append(f"TB 처음 보는 화자 val loss 상위 5: {b['val_unseen/loss']['top5']}")
            pick += [s for s in b["val_unseen/loss"]["top5"] if s not in pick]
    have = set(all_steps(run))
    pick = [s for s in pick if s in have][:5] or sorted(have)[-5:]
    return {"pick": pick, "lines": lines or ["(평가 전 — 저장된 마지막 5 step)"]}


def build(run: str, force: bool, out_dir: Path | None = None) -> Path:
    track = "lora" if "_LoRA_" in run else "sft"
    out = (out_dir or ROOT / "notebooks" / track) / f"{run}_listening.ipynb"
    if out.exists() and not force:
        print(f"🔒 이미 있음(덮어쓰지 않음): {out}")
        return out
    bi = best_info(run)
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python (voxcpm)", "language": "python", "name": "voxcpm"}
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    best_comment = "\n".join(f"#   · {l}" for l in bi["lines"])
    nb.cells = [
        md(f"""# 🎧 {run} 청음 — VoxCPM2 한국어 감정 FT

- 입력 규칙(D 재측정과 같음): **`(v1 지시문)문장` + 처음 보는 목표 화자의 무감정 reference** · prompt 없음 · 준비 합성 뒤 · seed 고정
- 무엇을 들을까: ① **목소리가 reference 와 같은 사람인가**(끝까지) ② **칸의 감정이 원본보다 분명한가**(A = 약 · B = 강) ③ 끊김 · 늘어짐 · 끝 잘림 · 발음
- 숫자(D 재측정 · TB)는 **참고용** — 귀가 정본. 비교 페이지: http://127.0.0.1:8765/vox_koft/d_eval/index.html
- 실행 순서: 1 → 2 → 3(ref) → 4(상위 5 step × 7칸) → `PICK_STEP` 고른 뒤 5 · 6 · 7"""),
        md("## 1. 환경"),
        code(f"""import os, sys
# ⚙️ GPU — 청음은 몇 GB 만 씀. 학습 중인 GPU 와 같이 써도 되지만 비어 있는 GPU 를 먼저.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
sys.path.insert(0, os.path.expanduser('~/workspaces/vox_koft'))
from voxko import listen
RUN_NAME = '{run}'
print('🎓', RUN_NAME, listen.run_info(RUN_NAME))
print('💾 저장된 step:', listen.steps(RUN_NAME))
print('LISTEN_SMOKE =', listen.SMOKE)"""),
        md("## 2. 노브 · ⭐ 들을 step"),
        code(f"""SEED, CFG, TIMESTEPS = 1986, 2.0, 10        # D 재측정과 같음
# ⭐ best 5 (참고용 — 귀로 확정)
{best_comment}
STEP_PICK_LIST = {bi['pick']}
PICK_STEP = STEP_PICK_LIST[0]              # ← 4절을 들은 뒤 귀로 고른 step 으로 바꾸고 이 셀부터 다시 실행
L = listen.Listener(RUN_NAME, seed=SEED, cfg_value=CFG, timesteps=TIMESTEPS)"""),
        md("## 3. Reference — 처음 보는 화자 16명(무감정) 중 고르기\n번호(`REF_IDX`)를 바꾸고 다시 실행하면 4절 이후가 그 목소리를 씁니다."),
        code("""listen.show_refs()
REF_IDX = 8                                 # 예: 8 = aihub_0012(남 · unseen8)
REF = listen.pick_ref(REF_IDX)
print('ref =', REF['speaker'], REF['gender'], REF['src'])"""),
        md("## 4. ⭐ 상위 5 step × 7칸 — 원본과 나란히\n칸마다 같은 문장 · 같은 ref · 같은 seed. 첫 실행은 step 을 바꿔 가며 합성합니다(step 하나 로드 약 10초)."),
        code("L.grid(steps=[None] + STEP_PICK_LIST, emotions=listen.EMO7, ref=REF)"),
        md("## 5. 고른 step(`PICK_STEP`) — 여러 목소리로\n여자 · 남자 처음 보는 화자 몇 명으로 같은 7칸을 들어 봅니다(목소리가 reference 를 따라가는지)."),
        code("""for i in [0, 4, 9, 12]:                     # 542 여 · 542 남 · unseen8 · unseen8
    L.grid(steps=[None, PICK_STEP], emotions=listen.EMO7, ref=listen.pick_ref(i))"""),
        md("## 6. 자유 문장 · 자유 지시문"),
        code("""TEXT = '정말 고마워요. 덕분에 오늘 하루가 완전히 달라졌어요!'
EMOTION = 'Happy-B'                          # 표의 감정 키(v1 문구) — 또는 INSTRUCT 에 직접
INSTRUCT = None                              # 예: 'slightly faster, cheerful tone'
from IPython.display import Audio, display
for st in [None, PICK_STEP]:
    w, sr, full = L.synth(st, TEXT, emotion=None if INSTRUCT else EMOTION, instruct=INSTRUCT, ref=REF)
    print('원본' if st is None else f'step {st}', '·', full); display(Audio(w, rate=sr))"""),
        md("## 7. seed 3개 반복 — 같은 입력에서 얼마나 흔들리나"),
        code("""for sd in ([1986] if listen.SMOKE else [1986, 1987, 1988]):
    L.grid(steps=[None, PICK_STEP], emotions=['Happy-B', 'Sad-B', 'Angry-B'], ref=REF, seed=sd)"""),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.validate(nb)
    nbf.write(nb, out)
    print(f"✅ {out} · STEP_PICK_LIST {bi['pick']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=None, help="시험용 출력 폴더(기본 notebooks/<sft|lora>/)")
    a = ap.parse_args()
    for r in a.runs:
        build(r, a.force, a.out_dir)


if __name__ == "__main__":
    main()
