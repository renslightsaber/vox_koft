#!/usr/bin/env python
"""학습 뒤 백업 · 종료 확인 — exp_reports/<sft|lora>/<run>/run_backup/ (q3tts_koft tools/post_train_backup.py 와 같은 역할)

    ~/miniforge3/envs/voxcpm/bin/python tools/post_train_backup.py A7N_ref_FT_01 [A7N_ref_LoRA_01_r32 …] [--out DIR]

만드는 것(run_backup/):
  logs/<원본>.log · logs/<원본>_readable.log(색 코드 뺌) · configs/(ckpt 의 실행 스냅샷 통째)
  tb_scalars.csv(모든 스칼라: tag · step · value · wall_time) · val_table.csv(step 별 val/* · val_unseen/*)
  analysis.json(종료 확인 · 저장 · 재개용 파일 · 경고 · best step · 학습 전 백업과 같은지) · md5.txt
그리고 exp_reports/<계열>/<run>/curves.png.
원본(ckpt · runs)은 읽기만 한다. 학습 중인 run 에 써도 안전(읽기만) — 그때는 "Training End 없음"으로 표시.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

H = Path.home() / "workspaces"
OUT = H / "outputs/vox_koft"
REPO = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def load_tb(run_tb: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out = defaultdict(dict)                                    # tag → {step: (value, wall)}
    for f in sorted(glob.glob(str(run_tb / "**/events.*"), recursive=True)):   # resume 로 여러 파일 → 뒤 파일이 같은 step 을 덮음
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for t in ea.Tags().get("scalars", []):
            for e in ea.Scalars(t):
                out[t][e.step] = (e.value, e.wall_time)
    return out


def body(p: Path) -> dict:
    return yaml.safe_load("\n".join(l for l in p.read_text(encoding="utf-8").splitlines() if not l.startswith("#"))) or {}


def curves(tb: dict, path: Path, run: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    panels = [("loss_diff", ["train/loss_diff", "val/loss_diff", "val_unseen/loss_diff"]),
              ("loss_stop", ["train/loss_stop", "val/loss_stop", "val_unseen/loss_stop"]),
              ("loss (sum)", ["train/loss", "val/loss", "val_unseen/loss"]),
              ("lr / grad_norm", ["train/lr", "train/grad_norm"])]
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    for ax, (title, tags) in zip(axs, panels):
        for t in tags:
            if t not in tb:
                continue
            s = sorted(tb[t])
            v = np.array([tb[t][k][0] for k in s])
            if t.startswith("train/") and t not in ("train/lr",) and len(v) > 20:
                k = 20                                      # 이동 평균(앞 k 개) — 양 끝을 0 으로 채우지 않음
                c = np.cumsum(np.insert(v, 0, 0.0))
                v = (c[k:] - c[:-k]) / k
                s = s[k - 1:]
            a2 = ax.twinx() if t == "train/grad_norm" else ax
            a2.plot(s, v, label=t, marker="o" if not t.startswith("train/") else None, ms=3,
                    color={"train": "#1f77b4", "val": "#ff7f0e", "val_unseen": "#2ca02c"}.get(t.split("/")[0], "#9467bd") if t != "train/grad_norm" else "#d62728")
            if t == "train/grad_norm":
                a2.set_ylabel("grad_norm")
        ax.set_title(f"{run} | {title}", fontsize=9)
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def backup(run: str, out_root: Path | None) -> dict:
    ck = OUT / "ckpt" / run
    if not ck.is_dir():
        sys.exit(f"ckpt 폴더가 없습니다: {ck}")
    merged = body(ck / "configs/merged.yaml")
    track = "lora" if (merged.get("lora") or {}).get("enable") else "sft"
    rep = (out_root or REPO / "exp_reports") / track / run
    rb = rep / "run_backup"
    rb.mkdir(parents=True, exist_ok=True)
    # 로그
    (rb / "logs").mkdir(exist_ok=True)
    warns, errs, end = [], [], None
    logs = sorted((ck / "logs").glob("*.log"))
    for lg in logs:
        shutil.copy2(lg, rb / "logs" / lg.name)
        txt = ANSI.sub("", lg.read_text(errors="ignore"))
        (rb / "logs" / (lg.stem + "_readable.log")).write_text(txt)
        for line in txt.splitlines():
            ln = line.strip()
            if re.search(r"Traceback|Error|❌|OutOfMemory", ln) and "_EXPORT_FAILED" not in ln:
                errs.append(f"{lg.name}: {ln[:200]}")
            elif re.search(r"⚠️| W ", ln):
                warns.append(f"{lg.name}: {ln[:200]}")
            m = re.search(r"Training End — step (\d+) · ([0-9:]+)", ln)
            if m:
                end = {"step": int(m.group(1)), "time": m.group(2), "log": lg.name}
    # 실행 스냅샷
    if (rb / "configs").exists():
        shutil.rmtree(rb / "configs")
    shutil.copytree(ck / "configs", rb / "configs")
    pre = rep / "config_backup" / "merged.yaml"
    same = (body(pre) == merged) if pre.exists() else None
    # 저장 상태
    steps = sorted((p for p in ck.glob("step_*") if p.is_dir()), key=lambda p: int(p.name.split("_")[1]))
    st = []
    for p in steps:
        c = json.loads((p / "_COMPLETE").read_text()) if (p / "_COMPLETE").exists() else None
        st.append({"step": int(p.name.split("_")[1]), "complete": c is not None, "resume": bool(c and c.get("resume")),
                   "resume_removed": (p / "_RESUME_REMOVED").exists(), "export_failed": (p / "_EXPORT_FAILED").exists(),
                   "exports": [e.name for e in p.glob("export_*") if e.is_dir()],
                   "has_state": (p / "optimizer.bin").exists()})
    # TensorBoard
    tb = load_tb(OUT / "runs" / run)
    with open(rb / "tb_scalars.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "step", "value", "wall_time"])
        for t in sorted(tb):
            for s in sorted(tb[t]):
                w.writerow([t, s, tb[t][s][0], tb[t][s][1]])
    vt = sorted(t for t in tb if t.startswith(("val/", "val_unseen/")))
    vsteps = sorted({s for t in vt for s in tb[t]})
    with open(rb / "val_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + vt)
        for s in vsteps:
            w.writerow([s] + [tb[t].get(s, ("",))[0] for t in vt])
    best = {}
    for t in ("val/loss", "val_unseen/loss", "val_unseen/loss_diff"):
        if t in tb and tb[t]:
            s = min(tb[t], key=lambda k: tb[t][k][0])
            best[t] = {"step": s, "value": tb[t][s][0], "top5": sorted(tb[t], key=lambda k: tb[t][k][0])[:5]}
    if tb:
        curves(tb, rep / "curves.png", run)
    wall = [v[1] for t in tb for v in tb[t].values()]
    a = {"run": run, "track": track, "training_end": end, "n_steps_saved": len(st),
         "n_complete": sum(x["complete"] for x in st), "n_resume_files": sum(x["has_state"] for x in st),
         "export_failed": [x["step"] for x in st if x["export_failed"]], "steps": st,
         "merged_same_as_pre_backup": same, "errors": errs[:50], "n_warnings": len(warns), "warnings": warns[:30],
         "tb_last_step": max((max(v) for v in tb.values()), default=None), "best": best,
         "wall_hours": round((max(wall) - min(wall)) / 3600, 2) if wall else None}
    (rb / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1))
    lines = [f"{md5(p)}  {p.relative_to(rb)}" for p in sorted(rb.rglob("*")) if p.is_file() and p.name != "md5.txt"]
    (rb / "md5.txt").write_text("\n".join(lines) + "\n")
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", type=Path, default=None, help="exp_reports 대신 다른 곳(시험용)")
    a = ap.parse_args()
    for r in a.runs:
        x = backup(r, a.out)
        print(json.dumps({k: x[k] for k in ("run", "training_end", "n_complete", "n_resume_files", "export_failed",
                                            "merged_same_as_pre_backup", "n_warnings", "tb_last_step", "wall_hours")}, ensure_ascii=False))
        print("   best:", {k: (v["step"], round(v["value"], 4)) for k, v in x["best"].items()}, "· errors", len(x["errors"]))


if __name__ == "__main__":
    main()
