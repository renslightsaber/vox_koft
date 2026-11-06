#!/usr/bin/env python
"""TensorBoard run 을 '학습 방식/데이터/run' 그룹 트리로 보이게 한다 (q3tts_koft tools/tb_group.py 를 옮김).

원본 runs/<run>/ 은 건드리지 않는다. runs_grouped/<방식>/<데이터>/ 는 진짜 폴더,
맨 끝 <run> 은 원본 run 폴더를 가리키는 **폴더 symlink** 하나다.
  · TensorBoard 2.21(env voxcpm)이 폴더 symlink 를 따라가고, resume 으로 생긴 새 기록 파일도 자동으로 읽는 것을
    2026-11-06 00:10 KST 에 직접 띄워 확인(파일 단위 링크는 새 기록 파일을 놓침 → 쓰지 않음). q3tts_koft 와 같은 결론.
다시 돌리면 새 run 만 추가하고, 원본이 사라진 깨진 링크 · 매핑이 바뀐 링크 · 빈 폴더는 정리한다(링크만 지움).

run 이름 규칙(docs/rules_in_this_proj/run_naming.md): 데이터 _ 조건 _ 학습 방식 _ 번호 _ 덧붙임
  학습 방식 = FT · LoRA (이름 끝이 _smoke 면 smoke) · 데이터 = A7N · A7N100 · A7N542 … · 안 맞으면 _misc/

새 학습 방식 이름은 method_of() 에 한 줄 추가(안 넣으면 _misc/) — 미리 확인: python tools/tb_group.py --where <run>
사용: python tools/tb_group.py [--dry-run] [--where RUN] [--src DIR] [--dst DIR]
안내: docs/guides/tensorboard_groups.md
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

OUT = Path.home() / "workspaces/outputs/vox_koft"
DATA_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")          # A7N · A7N100 · A7N542 …
COND_RE = re.compile(r"^[a-z][a-z0-9]*$")             # ref · noref · ref10 · inst0 …
METHODS = {"FT", "LoRA"}


def method_of(tok: str) -> str | None:
    """학습 방식 칸 → 묶음 이름. 새 방식(예: DPO)은 여기에 추가(q3tts_koft 와 같은 규칙 · 2026-11-06)."""
    return tok if tok in METHODS else None


def classify(run: str) -> tuple[str, str | None]:
    """run 이름 → (학습 방식, 데이터). 데이터 None 이면 2단(방식/run)."""
    parts = run.split("_")
    ok = len(parts) >= 3 and DATA_RE.match(parts[0]) and COND_RE.match(parts[1])
    if parts[-1] == "smoke":
        return "smoke", (parts[0] if ok else None)
    m = method_of(parts[2]) if ok else None
    if m:
        return m, parts[0]
    return "_misc", None


def group_path(run: str) -> Path:
    m, d = classify(run)
    return Path(m) / d / run if d else Path(m) / run


def sync(src: Path, dst: Path, dry: bool) -> None:
    runs = sorted(p.name for p in src.iterdir() if p.is_dir()) if src.is_dir() else []
    rows = [(r, group_path(r)) for r in runs]
    if dry:
        w = max([len(r) for r, _ in rows] or [1])
        for r, g in rows:
            print(f"{r:<{w}}  →  {g}")
        print()
        for k, n in sorted(Counter(str(g.parent) for _, g in rows).items()):
            print(f"{n:3d}  {k}")
        print(f"\n총 {len(rows)} run (dry-run, 아무것도 만들지 않음)")
        return
    added = removed = 0
    wanted: set[Path] = set()
    for run, g in rows:
        link, target = dst / g, (src / run).resolve()
        wanted.add(link)
        if link.is_symlink() and os.readlink(link) == str(target):
            continue
        if link.exists() and not link.is_symlink():
            sys.exit(f"진짜 폴더/파일이 있어 링크를 못 겁니다(직접 확인): {link}")
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target, target_is_directory=True)
        added += 1
    if dst.exists():                                    # 정리: 깨진 링크 · 지금 매핑에 없는 링크 · 빈 폴더(링크만 지움)
        for p in list(dst.rglob("*")):                  # rglob 은 폴더 링크 안으로 들어가지 않음
            if p.is_symlink() and p not in wanted:
                p.unlink()
                removed += 1
        for d in sorted((p for p in dst.rglob("*") if p.is_dir() and not p.is_symlink()),
                        key=lambda p: len(p.parts), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
    dst.mkdir(parents=True, exist_ok=True)              # run 이 아직 없어도 TensorBoard 가 띄워지게
    print(f"[tb_group] run {len(rows)} · 링크 추가 {added} · 정리 {removed} → {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=OUT / "runs")
    ap.add_argument("--dst", type=Path, default=OUT / "runs_grouped")
    ap.add_argument("--dry-run", action="store_true", help="매핑 표만 출력")
    ap.add_argument("--where", metavar="RUN", help="이 run 이 들어갈 그룹 경로만 출력(scripts/run.sh 가 시작 때 안내용으로 씀)")
    a = ap.parse_args()
    if a.where:
        print(group_path(a.where))
        return 0
    if a.dst.resolve() == a.src.resolve() or a.src.resolve() in a.dst.resolve().parents:
        sys.exit("dst 는 src 밖이어야 합니다")
    sync(a.src, a.dst, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
