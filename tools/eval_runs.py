#!/usr/bin/env python
"""본 학습 run 들의 D 재측정 일괄 실행 — GPU 여러 장에 나눠 합성 → 목소리 · 감정 · 깨짐 → 청음 페이지.

    ~/miniforge3/envs/voxcpm/bin/python tools/eval_runs.py --runs A7N_ref_FT_01 A7N_ref_LoRA_01_r32 … --gpus 2 3 [--steps 750 1500 2250 3000] [--dry]

step 고르기(run 마다): --steps(기본 750 · 1500 · 2250 · 3000) + TensorBoard 처음 보는 화자 val loss 가 가장 낮은 step(없으면 생략).
  → 저장된 step 전부(12개 × 4 run = 48 모델 · 약 4시간)는 밤사이에 다 못 돌려서 줄임(이유는 보고서에 기록).
이름표 = <짧은 run 이름>_s<step> (FT01 · LoRA01 · FT02iv2 · LoRA02iv2). 원본 = base.
단계(각 단계는 맞는 env 로 voxko/eval/d_eval.py 를 부름)
  1) vox    (voxcpm)    : 이름표를 GPU 수만큼 나눠 동시에 합성 — 이미 있는 wav 는 건너뜀
  2) sv     (q3tts)     : 목소리 두 잣대 — GPU 0번째 · base 도 ERes2Net 잣대가 없으면 다시
  2') quality(voxcpm)   : 깨짐 — GPU 1번째(sv 와 동시에)
  3) ser    (q3tts_ser) : emotion2vec · arousal · whisper CER — 이름표를 GPU 수만큼 나눠 동시에
  4) page   : 청음 열 = base + run 마다 참고 점수 1등 step(요약 표는 전부)
학습 프로세스가 GPU 에 남아 있으면 시작하지 않는다(--force 로 무시).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = Path.home() / "workspaces"
OUT = H / "outputs/vox_koft"
D = H / "tmp_listen/vox_koft/d_eval"
ENV = {k: H.parent / f"miniforge3/envs/{v}/bin/python" for k, v in (("vox", "voxcpm"), ("q3", "q3tts"), ("ser", "q3tts_ser"))}
SHORT = {"A7N_ref_FT_01": "FT01", "A7N_ref_LoRA_01_r32": "LoRA01", "A7N_ref_FT_02_iv2": "FT02iv2", "A7N_ref_LoRA_02_r32_iv2": "LoRA02iv2",
         "A7NX_xref_LoRA_01_r32": "A7NX", "A7NX3_xref_LoRA_01_r32": "A7NX3",                       # 2026-11-07 다른 칸 ref
         "A7NX2R_xref_LoRA_01_r32": "X2R01", "A7NX2R_xref_LoRA_02_r32_lr5e5": "X2R02lr5e5",
         "A7NX2Rs_xref_LoRA_01_r32_lr5e5": "X2Rs01lr5e5",
         "A7NX2Rs6_xref_LoRA_01_r32_lr5e5": "X2Rs6lr5e5",
         "A7NUX2Rs6_xref_LoRA_01_r32_lr5e5": "NUX2Rs6lr5e5", "A7NUX2s_xref_LoRA_01_r32_lr5e5": "NUX2slr5e5",
         "A7NUX2sN_xref_LoRA_01_r32_lr5e5": "NUX2sNlr5e5",  # 11/07 18:40 · 21:45
         "A7NUX2sN_xref_LoRA_02_r32_lr5e5_iv3": "NUX2sNiv3lr5e5", "A7NUX2sNn_xref_LoRA_01_r32_lr5e5": "NUX2sNnlr5e5"}  # 11/09 (iv3 는 v3 문구로 잼 · v1 문구 참고는 --short NUX2sNiv3v1)
DE = str(ROOT / "voxko/eval/d_eval.py")


def log(m):
    print(time.strftime("[%H:%M:%S UTC] ") + m, flush=True)


def best_unseen(run: str):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import glob
        v = {}
        for f in sorted(glob.glob(str(OUT / "runs" / run / "**/events.*"), recursive=True)):
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
            if "val_unseen/loss" in ea.Tags()["scalars"]:
                v.update({e.step: e.value for e in ea.Scalars("val_unseen/loss")})
        return min(v, key=v.get) if v else None
    except Exception as e:  # noqa: BLE001
        log(f"TB 읽기 실패({run}): {e}")
        return None


def export_of(run, step):
    d = OUT / "ckpt" / run / f"step_{step}"
    for k in ("export_ft", "export_lora"):
        if (d / k).is_dir() and (d / "_COMPLETE").exists():
            return d / k
    return None


def gpus_busy(gpus) -> list:
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    uu = {l.split(", ")[1].strip(): int(l.split(", ")[0]) for l in q.strip().splitlines()}
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [(int(l.split(", ")[0]), uu.get(l.split(", ")[1].strip())) for l in a.strip().splitlines() if l and uu.get(l.split(", ")[1].strip()) in gpus]


def run_parallel(cmds_by_gpu: dict, logdir: Path, name: str):
    """GPU 마다 명령 목록을 차례로 — GPU 끼리는 동시에."""
    procs = {}
    for g, cmds in cmds_by_gpu.items():
        if not cmds:
            continue
        script = " && ".join(" ".join(map(str, c)) for c in cmds)
        lf = open(logdir / f"{name}_gpu{g}.log", "a")
        procs[g] = subprocess.Popen(["bash", "-c", script], env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(g)},
                                    stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
    rc = {g: p.wait() for g, p in procs.items()}
    log(f"{name} 끝 · rc {rc}")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gpus", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--steps", nargs="+", type=int, default=[750, 1500, 2250, 3000])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=["vox", "score", "page"], default=None)
    ap.add_argument("--emotion_map", default=None, help="합성 지시문 표(기본 v1) — 예: configs/emotion/a7n_vox_v3.yaml")
    ap.add_argument("--short", default=None, help="이름표 앞부분을 이것으로(run 1개일 때 · 예: NUX2sNiv3v1 = 같은 ckpt 를 v1 문구로)")
    a = ap.parse_args()
    busy = gpus_busy(a.gpus)
    if busy and not a.force:
        sys.exit(f"GPU {a.gpus} 에 다른 프로세스가 있습니다 {busy} — 학습이 끝난 뒤 실행(--force 로 무시)")
    jobs = []                                                             # (tag, model)
    for r in a.runs:
        st = list(a.steps)
        b = best_unseen(r)
        if b and b not in st:
            st.append(b)
        for s in sorted(st):
            e = export_of(r, s)
            if e is None:
                log(f"⚠️ {r} step {s}: export 없음(건너뜀)")
                continue
            jobs.append((f"{(a.short if a.short else SHORT.get(r, r))}_s{s}", e))
        log(f"{r}: 평가 step {sorted(st)} (TB val_unseen 최저 {b})")
    tags = [t for t, _ in jobs]
    (D / "eval_runs_plan.json").write_text(json.dumps({"runs": a.runs, "jobs": [[t, str(m)] for t, m in jobs]}, indent=1))
    log(f"모델 {len(jobs)}개: {tags}")
    if a.dry:
        return
    L = D / "logs"
    L.mkdir(exist_ok=True)
    G = a.gpus
    if a.only in (None, "vox"):
        by = {g: [] for g in G}
        todo = [(t, m) for t, m in jobs if not (D / f"vox_{t}.json").exists()]
        for k, (t, m) in enumerate(todo):
            by[G[k % len(G)]].append([ENV["vox"], DE, "vox", "--tag", t, "--model", m] + (["--emotion_map", a.emotion_map] if a.emotion_map else []))
        log(f"1) 합성 {len(todo)}개 → GPU {G}")
        run_parallel(by, L, "vox")
    if a.only in (None, "score"):
        all_tags = ["base"] + tags
        log("2) 목소리(sv) · 깨짐(quality) 동시에")
        run_parallel({G[0]: [[ENV["q3"], DE, "sv", "--tags", *all_tags]],
                      G[-1] if len(G) > 1 else G[0]: [[ENV["vox"], DE, "quality", "--tags", *all_tags]]}, L, "sv_quality")
        log("3) 감정 · CER(ser)")
        by = {g: [] for g in G}
        for k, t in enumerate(all_tags):
            by[G[k % len(G)]].append(t)
        run_parallel({g: [[ENV["ser"], DE, "ser", "--tags", *ts]] for g, ts in by.items() if ts}, L, "ser")
    if a.only in (None, "page", "score"):
        sys.path.insert(0, str(ROOT))
        subprocess.run([ENV["vox"], DE, "page"], cwd=ROOT, check=False)       # 요약을 만들어 1등 step 을 고름
        summ = json.load(open(D / "summary.json"))["models"]
        pt = ["base"]
        for r in a.runs:
            pre = (a.short if a.short else SHORT.get(r, r)) + "_s"
            c = [t for t in summ if t.startswith(pre)]
            if c:
                pt.append(max(c, key=lambda t: summ[t]["전체"]["score"]))
        log(f"4) 청음 열: {pt}")
        subprocess.run([ENV["vox"], DE, "page", "--page_tags", *pt], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
