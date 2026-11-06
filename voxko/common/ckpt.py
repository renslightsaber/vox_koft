"""체크포인트 · 실행 정책 · 설정 스냅샷.

체크포인트 = accelerate save_state() 파일들 + export_<형식>/ + `_COMPLETE` 표식(step·epoch·시각).
export 가 실패하면 `_COMPLETE` 대신 `_EXPORT_FAILED`(오류 내용)를 남긴다 — 학습 상태는 그대로라 resume 은 된다.

cv3_koft(src/cv3kft/common/checkpoint.py) 에서 이식·개선한 것
  · 재진입·완주·resume 판단을 **모델·TensorBoard 를 만들기 전에** 한 곳(decide_run_policy)에서 끝낸다.
    (이전: 가드보다 TB 초기화가 먼저라, 거부된 실행이 기존 run 의 TB 폴더에 이벤트 파일을 남겼다)
  · smoke 는 **절대 resume 하지 않는다**(`--smoke --resume_from_latest` 가 0 step 으로 끝나던 버그).
  · 스냅샷: fresh 는 configs/ 를 새로(이전 실패 시도는 configs_prev_<ts>/ 로), resume 은 cmd_resume_<ts>.txt
    + merged_resume_<ts>.yaml 을 **추가**(무엇을 바꿔 재개했는지 남는다).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import List, Optional

COMPLETE_MARK = "_COMPLETE"
RESUME_REMOVED_MARK = "_RESUME_REMOVED"   # 재개용 파일(optimizer 등)을 지우고 export_* 만 남긴 step 표시
EXPORT_FAILED_MARK = "_EXPORT_FAILED"
_STEP_RE = re.compile(r"^step_(\d+)$")
# accelerate save_state(safe_serialization=True) 산출물. custom_checkpoint_* = TrainState·scheduler
_REQUIRED = ("model.safetensors", "optimizer.bin", "random_states_0.pkl",
             "custom_checkpoint_0.pkl", "custom_checkpoint_1.pkl")


class TrainState:
    """체크포인트에 함께 저장되는 학습 상태 (accelerator.register_for_checkpointing).

    batch_in_epoch: 이번 epoch 에서 이미 소비한 micro-batch 수(rank 마다 같음) → resume 때 정확히 그 다음부터.
    max_steps·sched: 저장 당시 스케줄 — resume 때 바뀌었으면 경고/중단(cosine 곡선이 조용히 튀는 것 방지).
    """

    def __init__(self):
        self.global_step, self.epoch, self.batch_in_epoch = 0, 0, 0
        self.max_steps: Optional[int] = None
        self.sched: dict = {}
        self.seed: Optional[int] = None
        # 추가 학습 데이터(data.extra_sources) 의 위치: {name: {"epoch", "batch_in_epoch", "per_step"}} (2026-10-28)
        # 원래 데이터와 따로 돈다. 비어 있으면 저장하지 않는다(기존 체크포인트와 같은 내용).
        self.extra: dict = {}

    def state_dict(self):
        d = {"global_step": self.global_step, "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
             "max_steps": self.max_steps, "sched": self.sched, "seed": self.seed}
        if self.extra:
            d["extra"] = {k: dict(v) for k, v in self.extra.items()}
        return d

    def load_state_dict(self, d):
        self.global_step, self.epoch = int(d.get("global_step", 0)), int(d.get("epoch", 0))
        self.batch_in_epoch = int(d.get("batch_in_epoch", 0))
        self.max_steps, self.sched, self.seed = d.get("max_steps"), d.get("sched") or {}, d.get("seed")
        self.extra = {k: dict(v) for k, v in (d.get("extra") or {}).items()}


# ── 탐색 · 회전 ──────────────────────────────────────────────────────────────
def step_dirs(root: str | Path) -> List[Path]:
    """root 아래 step_<N> 디렉터리를 step 번호 오름차순으로."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in root.iterdir() if p.is_dir() and (m := _STEP_RE.match(p.name))]
    return [p for _, p in sorted(found)]


def step_of(path: str | Path) -> int:
    m = _STEP_RE.match(Path(path).name)
    return int(m.group(1)) if m else -1


def is_complete(path: str | Path) -> bool:
    """학습 상태가 끝까지 저장됐는가(_COMPLETE, 없으면 필수 파일이 다 있고 0바이트가 아닌지)."""
    p = Path(path)
    if (p / COMPLETE_MARK).exists():
        return True
    return all((p / f).exists() and (p / f).stat().st_size > 0 for f in _REQUIRED)


def mark_complete(path: str | Path, **info) -> None:
    info.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    (Path(path) / COMPLETE_MARK).write_text(json.dumps(info, ensure_ascii=False) + "\n")


def latest_checkpoint(root: str | Path, warn=None) -> Optional[Path]:
    """가장 최신의 완전한 체크포인트. 손상본을 건너뛰면 warn(msg) 으로 알린다."""
    dirs = step_dirs(root)
    for p in reversed(dirs):
        if is_complete(p):
            if p != dirs[-1] and warn:
                warn(f"⚠️ 최신 {dirs[-1].name} 가 불완전합니다(저장 중 중단 추정) — {p.name} 에서 이어갑니다.")
            return p
    return None


def has_resume_files(path: str | Path) -> bool:
    """이 step 에서 이어서 학습할 수 있는가 = 완전한 체크포인트 · 재개용 파일을 지우지 않음 · export 만 저장한 step 이 아님.

    export 만 저장한 step(train.resume.every 사이의 save_every)은 `_COMPLETE` 에 "resume": false 가 적혀 있다(2026-11-04).
    옛 체크포인트(_COMPLETE 에 resume 키 없음)는 재개용으로 본다.
    """
    p = Path(path)
    if not is_complete(p) or (p / RESUME_REMOVED_MARK).exists():
        return False
    mark = p / COMPLETE_MARK
    if mark.exists():
        try:
            if json.loads(mark.read_text() or "{}").get("resume") is False:
                return False
        except Exception:  # noqa: BLE001 — 옛 표식 형식은 재개용으로 본다
            pass
    return True


def latest_resumable(root: str | Path) -> Optional[Path]:
    """재개용 파일이 온전한 가장 큰 step(없으면 None)."""
    for p in reversed(step_dirs(root)):
        if has_resume_files(p):
            return p
    return None


RESUME_KEEP_NAMES = (COMPLETE_MARK, EXPORT_FAILED_MARK, RESUME_REMOVED_MARK)


def remove_resume_files(step_dir: str | Path, why: str) -> List[str]:
    """step 폴더에서 재개용 파일만 지우고(export_*/ · 표식은 남김) `_RESUME_REMOVED` 를 남긴다. 지운 이름 반환."""
    p = Path(step_dir)
    gone = []
    for f in sorted(p.iterdir()):
        if f.name in RESUME_KEEP_NAMES or (f.is_dir() and f.name.startswith("export_")):
            continue
        shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink(missing_ok=True)
        gone.append(f.name)
    (p / RESUME_REMOVED_MARK).write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "removed": gone,
                                                     "kept": "export_*/ · _COMPLETE", "why": why},
                                                    ensure_ascii=False) + "\n")
    return gone


def prune_resume(root: str | Path, keep: int, protect: Optional[str | Path] = None) -> List[Path]:
    """재개용 파일이 있는 step 중 **가장 큰 keep 개**만 남기고 나머지 step 의 재개용 파일을 지운다(export 는 그대로)."""
    keep = max(1, int(keep))
    have = [p for p in step_dirs(root) if has_resume_files(p)]
    protect = Path(protect).resolve() if protect else None
    victims = [p for p in have[:max(0, len(have) - keep)] if p.resolve() != protect]
    for p in victims:
        remove_resume_files(p, f"train.resume.keep={keep} — 최근 {keep}개만 재개용으로 보관")
    return victims


def rotate(root: str | Path, keep_last: Optional[int], protect: Optional[str | Path] = None) -> List[Path]:
    """**step 번호가 큰** keep_last 개만 남긴다(protect 는 지우지 않음). 지운 목록 반환.

    keep_last: None·음수 = 아무것도 안 지움, 0 = protect 외 전부 삭제(저장 직전 정리에서 keep_last=1 일 때).
    재개용 파일이 있는 가장 최근 step 도 지우지 않는다(export 만 저장하는 step 이 늘어도 이어 하기가 가능하게).
    """
    if keep_last is None or keep_last < 0:
        return []
    dirs = step_dirs(root)
    keep = {Path(protect).resolve()} if protect else set()
    res = latest_resumable(root)                     # 가장 최근 재개용 step 은 회전으로 지우지 않는다(2026-11-04)
    if res is not None:
        keep.add(res.resolve())
    victims = [d for d in dirs[:max(0, len(dirs) - keep_last)] if d.resolve() not in keep]
    for d in victims:
        shutil.rmtree(d, ignore_errors=True)
    return victims


# ── 실행 정책 (모델 빌드 전에) ───────────────────────────────────────────────
def decide_run_policy(run_dir: str | Path, max_steps: int, resume: bool, force_overwrite: bool,
                      smoke: bool, warn=None) -> dict:
    """이번 실행을 smoke / fresh / resume 중 무엇으로 할지 정한다. 진행하면 안 되면 SystemExit.

    모든 rank 가 같은 파일을 보고 같은 결론을 낸다(메시지 출력은 호출부가 rank0 에서만).
    반환: {"mode": "smoke"|"fresh"|"resume", "resume_from": Path|None}
    """
    if smoke:
        if resume and warn:
            warn("[smoke] --resume_from_latest 는 smoke 에서 무시합니다(smoke 는 저장도 재개도 하지 않음).")
        return {"mode": "smoke", "resume_from": None, "stale": []}
    existing = step_dirs(run_dir)
    if resume:
        latest = latest_checkpoint(run_dir, warn)
        if latest is None:
            if existing:
                raise SystemExit(f"❌ {run_dir} 의 체크포인트가 전부 불완전합니다 — 직접 확인하세요.")
            if warn:
                warn(f"[resume] {run_dir} 에 체크포인트가 없어 처음부터 시작합니다.")
            return {"mode": "fresh", "resume_from": None, "stale": []}
        if step_of(latest) >= max_steps:
            raise SystemExit(f"⏭  이미 완주한 run 입니다 — {latest.name} ≥ max_steps={max_steps}.\n"
                             f"   · 더 돌리려면 train.max_steps 를 올리고 --allow_schedule_change 를 주세요"
                             f"(스케줄 곡선이 바뀝니다).\n   · 처음부터 다시 하려면 다른 실험 파일명(run_name)을 쓰세요.")
        res = latest_resumable(run_dir)
        if res is None:
            raise SystemExit(f"⛔ {run_dir} 는 재개(resume)용 파일이 남은 step 이 없습니다 "
                             f"(최신 {latest.name}: export 만 저장했거나 {RESUME_REMOVED_MARK}).\n"
                             f"   · 추론·청음용 export 만 남아 있어 이어서 학습할 수 없습니다.\n"
                             f"   · 처음부터 다시 하려면 다른 실험 파일명(run_name)을 쓰세요.")
        stale = [d for d in existing if step_of(d) > step_of(res)]
        if stale and warn:
            warn(f"[resume] 재개용 파일이 있는 마지막 step = {res.name} — 그 뒤 step({', '.join(d.name for d in stale)})은 "
                 f"export 만 있으므로 지우고 다시 만듭니다.")
        return {"mode": "resume", "resume_from": res, "stale": stale}
    if existing and not force_overwrite:
        raise SystemExit(f"🔴 {run_dir} 에 이미 체크포인트가 {len(existing)}개 있습니다 (최신 {existing[-1].name}).\n"
                         f"   → 이어서 하려면 : --resume_from_latest\n"
                         f"   → 덮어쓰려면    : --force_overwrite (기존 step_* 와 섞이므로 권장하지 않음)\n"
                         f"   → 새로 하려면   : 다른 실험 파일명(run_name)")
    return {"mode": "fresh", "resume_from": None, "stale": []}


# ── 설정 스냅샷 ──────────────────────────────────────────────────────────────
def md5_head(path: str, n: int = 12) -> str:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:n]
    except OSError:
        return "missing"


def physical_gpu(device: int = 0) -> Optional[int]:
    """이 프로세스의 cuda:device 가 nvidia-smi 의 몇 번 GPU 인지(UUID 대조). 알 수 없으면 None."""
    import subprocess

    import torch
    try:
        uuid = "GPU-" + str(torch.cuda.get_device_properties(device).uuid)
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            idx, u = [x.strip() for x in line.split(",")]
            if u == uuid:
                return int(idx)
    except Exception:  # noqa: BLE001
        pass
    return None


def _cmd_text(cfg: dict, mode: str, world: int, seed: int) -> str:
    import torch
    d = cfg["_derived"]
    lines = [f"# cmd.txt — run {d['run_name']} ({mode})",
             f"time        : {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
             f"host        : {socket.gethostname()}  user={os.environ.get('USER', '?')}",
             f"conda env   : {os.environ.get('CONDA_DEFAULT_ENV', '?')}",
             f"world size  : {world}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(all)')}",
             f"seed        : {seed}",
             f"alloc conf  : PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(없음)')}"]
    if torch.cuda.is_available():
        lines.append(f"gpu         : {', '.join(sorted({torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}))}")
        lines.append(f"physical gpu: {[physical_gpu(i) for i in range(torch.cuda.device_count())]}  (nvidia-smi 번호)")
    try:
        import accelerate
        import transformers
        lines.append(f"versions    : torch {torch.__version__} · transformers {transformers.__version__} · "
                     f"accelerate {accelerate.__version__}")
    except Exception:  # noqa: BLE001
        pass
    for k in ("train_jsonl", "val_jsonl"):
        p = d.get(k)
        n = sum(1 for _ in open(p, encoding="utf-8")) if p and os.path.isfile(p) else 0
        lines.append(f"{k:<12}: {p}  (rows={n}, md5={md5_head(p) if p else '-'})")
    lines += [f"emotion map : {d.get('emotion_map')}  (md5={md5_head(d['emotion_map']) if d.get('emotion_map') else '-'})",
              f"init_model  : {d.get('init_model_dir')}",
              f"launcher    : {os.environ.get('VOX_LAUNCH_CMD', '(직접 실행)')}",
              f"accel file  : {os.environ.get('VOX_ACCEL_FILE', '(알 수 없음 — scripts/run.sh 경유 시 기록)')}",
              f"argv        : {' '.join(sys.argv)}"]
    if cfg["_run"].get("sets"):
        lines.append(f"--set       : {' '.join(cfg['_run']['sets'])}")
    return "\n".join(lines) + "\n"


def write_snapshot(cfg: dict, mode: str, world: int, seed: int) -> Path:
    """{run_dir}/configs/ 에 설정 기록(rank0 에서만 호출).

    fresh  : configs/ 새로 — merged.yaml(최종 병합본) + 입력 yaml 원본 + accelerate 파일 + cmd.txt.
             이미 있으면(= 체크포인트 없이 끝난 이전 시도) configs_prev_<ts>/ 로 옮긴다.
    resume : 기존 configs/ 보존 + cmd_resume_<ts>.txt + merged_resume_<ts>.yaml 추가.
    """
    import yaml

    from .config import public_view
    snap = Path(cfg["_derived"]["run_dir"]) / "configs"
    ts = time.strftime("%Y%m%d_%H%M%S")
    dump = lambda f: yaml.safe_dump(public_view(cfg), f, allow_unicode=True, sort_keys=False)  # noqa: E731
    if mode == "resume" and snap.is_dir():
        (snap / f"cmd_resume_{ts}.txt").write_text(_cmd_text(cfg, mode, world, seed), encoding="utf-8")
        with open(snap / f"merged_resume_{ts}.yaml", "w", encoding="utf-8") as f:
            dump(f)
        return snap
    if snap.is_dir():
        snap.rename(snap.with_name(f"configs_prev_{ts}"))
    snap.mkdir(parents=True)
    for role, src in cfg["_run"]["sources"].items():
        shutil.copy2(src, snap / f"{role}__{Path(src).name}")
    em = cfg["_derived"].get("emotion_map")               # 감정 지시문 표도 함께 보관(학습 때 붙이므로 run 의 일부)
    if em and os.path.isfile(em):
        shutil.copy2(em, snap / f"emotion_map__{Path(em).name}")
    acc_file = os.environ.get("VOX_ACCEL_FILE")
    if acc_file and os.path.isfile(acc_file):
        shutil.copy2(acc_file, snap / f"accelerate__{Path(acc_file).name}")
    with open(snap / "merged.yaml", "w", encoding="utf-8") as f:
        f.write("# 이 run 이 실제로 쓴 최종 설정(config → hardware → experiment → overlay → --set 병합). 🔒 수정 금지\n")
        dump(f)
    (snap / "cmd.txt").write_text(_cmd_text(cfg, mode, world, seed), encoding="utf-8")
    return snap


# ── 파일 링크 ────────────────────────────────────────────────────────────────
def link_or_copy(src: Path, dst: Path) -> None:
    """하드링크(같은 디스크면 용량 0), 안 되면 복사. 디렉터리는 재귀."""
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            link_or_copy(p, dst / p.name)
        return
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
