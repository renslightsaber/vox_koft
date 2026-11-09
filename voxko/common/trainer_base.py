"""BaseTrainer — accelerate 학습 인프라 (모델에 무관한 부분). 전체 FT · LoRA 가 서브클래스로 재사용한다.

q3tts_koft/q3ko/common/trainer_base.py 를 옮긴 것(2026-11-05). 바뀐 점: 지표 이름(loss_diff · loss_stop) ·
오디오 샘플레이트(정답 16kHz · 합성 48kHz) · 샘플러 위치(voxko.common.sampler) · 환경변수 VOX_*.

실행 순서
  ① 실행 정책(fresh/resume/smoke/거부)  ← 모델·TensorBoard 를 만들기 **전에** (거부된 실행이 흔적을 남기지 않게)
  ② Accelerator · expected_num_processes 검사 · seed
  ③ 훅: build_model() · build_datasets() · preflight()  → seed 재적용(빌드 중 난수 소비와 무관하게)
  ④ 샘플러·로더 · optimizer(decay/no-decay) · scheduler(LambdaLR, 수동 step) · prepare(model, optimizer)
  ⑤ resume: load_state → 스케줄·optimizer 하이퍼 정합 → TrainState 로 데이터 위치 정확 복원
  ⑥ TensorBoard(smoke 제외) · 설정 스냅샷(rank0)
  ⑦ fit(): accumulation 창 단위 루프 → log / eval / media / save

서브클래스가 구현할 훅
  build_model() -> nn.Module          accelerate.prepare 대상(forward(batch, ...) 가 합계 dict 를 반환)
  build_datasets() -> (train, val)    .lengths · .collate · collate 결과에 "counts"(정규화용 개수 텐서)
  forward_step(model, batch) -> dict  합계 텐서들(sum_keys)
  loss_from_sums(out, denom) -> Tensor  denom = 창·전 rank 의 counts 합 → 모든 token 같은 가중치
  metrics_from_sums(sums) -> dict     로그·val 지표
  export(step_dir) -> list[(fmt, path)] / verify_export(fmt, path)
  media_rows() · synthesize(rows) -> list[np.ndarray]
  banner_sections() · describe_data() · preflight()

원형: ~/workspaces/q3tts_koft/q3ko/common/trainer_base.py
"""
from __future__ import annotations

# ★ console 을 가장 먼저 — colorama → logging → tqdm 순서 고정
from voxko.common import console
from voxko.common.console import c, fmt_metrics, fmt_seconds, logger, logging_redirect_tqdm, tqdm

import itertools
import math
import os
import json
import shutil
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from voxko.common import ckpt
from voxko.common.audio import SR_IN, SR_OUT, load_audio, mel, mel_to_figure
from voxko.common.config import flatten
from voxko.common.rng import preserved_rng

_NO_DECAY = ("bias", "norm", "Norm")


def lr_lambda(name: str, warmup: int, total: int):
    """warmup 첫 step 의 배수는 1/warmup(0 이 아님) — 1-step smoke 에서도 가중치가 실제로 움직인다."""
    if name == "constant_warmup":
        return lambda s: min(1.0, (s + 1) / max(warmup, 1))
    if name == "cosine_warmup":
        def f(s):
            if s < warmup:
                return (s + 1) / max(warmup, 1)
            prog = (s - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
        return f
    raise ValueError(f"optim.scheduler 는 constant_warmup | cosine_warmup: {name!r}")


def _stop(msg: str) -> None:
    """모든 rank 가 같은 판단으로 멈출 때: 메시지는 rank0 만 출력하고 종료 코드 1."""
    console.say(msg, "RED", "BRIGHT")
    raise SystemExit(1)


class BaseTrainer:
    title = "Trainer"
    sum_keys: Tuple[str, ...] = ()

    def __init__(self, cfg: dict, smoke: bool = False, resume: bool = False, force_overwrite: bool = False,
                 allow_schedule_change: bool = False):
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed

        self.cfg, self.smoke = cfg, smoke
        d, tc, oc, dc = cfg["_derived"], cfg["train"], cfg["optim"], cfg["data"]
        self.mc, self.ec = cfg.get("media", {}), cfg.get("export", {})
        self.run_name, self.run_dir, self.tb_dir = d["run_name"], Path(d["run_dir"]), d["tb_dir"]
        self.max_steps = int(tc["smoke_steps"] if smoke else tc["max_steps"])
        self.grad_accum = int(tc["grad_accum"])
        self.grad_clip = float(oc["grad_clip"])
        self.log_every, self.eval_every, self.save_every = int(tc["log_every"]), int(tc["eval_every"]), int(tc["save_every"])
        self.media_every = int(self.mc.get("every", 0)) if self.mc.get("enable", True) else 0
        self.eval_max_batches = 2 if smoke else tc.get("eval_max_batches")
        # val 전체를 평가하면 token 가중 합계라 배치 구성과 무관하게 같은 값이 나온다. 상한을 두면 표본이
        # 배치 설정(batch_size·max_tokens — hardware yaml 이 바꿀 수 있음)에 따라 달라진다 → 실험 간 비교 불가.
        self._eval_cap_warning = (not smoke and self.eval_max_batches is not None)
        self.keep_last, self.seed = int(tc["keep_last"]), int(tc["seed"])
        # 재개(resume)용 파일 저장(2026-11-04): export_base 는 save_every 마다 그대로, 재개용 파일(optimizer 등 · 약 3.5GB)은
        # resume.every 마다(+ 마지막 step) 저장하고 가장 최근 resume.keep 개만 남긴다. enable=false 면 재개용 파일을 안 남긴다.
        rc = tc.get("resume") or {}
        self.resume_enable = bool(rc.get("enable", True))
        self.resume_every = int(rc.get("every") or 0)
        self.resume_keep = int(rc.get("keep") or 1)
        self.early_traj = set(int(s) for s in (tc.get("early_traj_steps") or []))
        self.use_tb = bool(cfg["logging"]["tensorboard"]) and not smoke
        # 추가 학습 데이터(data.extra_sources, 2026-10-28): 창(grad_accum 개 micro-batch) 끝에 소스마다
        # per_step / (batch_size × GPU) 개 micro-batch 를 더한다 → accelerate 의 누적 길이 = window. 없으면 window = grad_accum(기존과 같음)
        world = int(cfg["hardware"]["expected_num_processes"] or 1)
        self.extra_micro = sum(int(e["per_step"]) // (int(dc["batch_size"]) * world) for e in (dc.get("extra_sources") or []))
        self.window = self.grad_accum + self.extra_micro

        # ── ① 실행 정책: 무거운 것을 만들기 전에 ──
        try:
            self.policy = ckpt.decide_run_policy(self.run_dir, self.max_steps, resume, force_overwrite, smoke,
                                                 warn=lambda m: console.say(m, "YELLOW"))
        except SystemExit as e:                               # 메시지는 rank0 만, 종료는 모든 rank 가 함께
            console.say(str(e.code), "RED", "BRIGHT")
            raise SystemExit(1) from None
        self.mode = self.policy["mode"]
        if self.policy.get("stale") and os.environ.get("RANK", "0") == "0":   # 재개 지점 뒤의 export 만 있는 step → 다시 만듦
            for sd in self.policy["stale"]:
                shutil.rmtree(sd, ignore_errors=True)

        # ── ② accelerate ──
        self.acc = acc = Accelerator(
            gradient_accumulation_steps=self.window, mixed_precision="bf16",
            log_with="tensorboard" if self.use_tb else None, project_dir=d["tb_root"],
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=60)),   # rank0 media·export 중 대기
                             DistributedDataParallelKwargs(find_unused_parameters=bool(tc["ddp_find_unused"]))])
        console.set_main_process(acc.is_main_process)
        want = int(cfg["hardware"]["expected_num_processes"])
        if acc.num_processes != want:
            _stop(f"❌ GPU 프로세스 수 불일치: accelerate={acc.num_processes} · "
                             f"hardware.expected_num_processes={want}\n   accelerate 파일과 hardware 파일의 짝을 맞추세요"
                             f"(scripts/run.sh 를 쓰면 자동).")
        expect = os.environ.get("VOX_EXPECT_GPUS")              # scripts/run.sh 가 설정(예: "2", "2,3")
        if expect and torch.cuda.is_available():
            phys = ckpt.physical_gpu(acc.local_process_index)
            if phys is not None and phys not in {int(x) for x in expect.split(",")}:
                _stop(f"❌ 실제 GPU 가 지정과 다릅니다: rank{acc.process_index} → GPU {phys} (지정 {expect})\n"
                      f"   accelerate 파일의 gpu_ids 가 CUDA_VISIBLE_DEVICES 를 덮어쓰는지 확인하세요.")
            console.say(f"  GPU 확인: rank{acc.process_index} → nvidia-smi {phys}번 (지정 {expect})")
        set_seed(self.seed)

        # ── ③ 모델·데이터 (훅) ──
        self.module = self.build_model()
        self.ds_train, self.ds_val = self.build_datasets()
        self.preflight()
        set_seed(self.seed)                                   # 빌드 중 난수 소비와 무관하게 학습 시작 시점 고정

        # ── ④ 로더 · optimizer · scheduler ──
        from voxko.common.sampler import BucketBatchSampler, EvalBatchSampler
        bs = dict(batch_size=int(dc["batch_size"]), max_tokens=dc.get("max_tokens"),
                  num_replicas=acc.num_processes, rank=acc.process_index, bucket_mult=int(dc["bucket_mult"]))
        self.sampler = BucketBatchSampler(self.ds_train.lengths, seed=self.seed, multiple_of=self.grad_accum,
                                          tag_epoch=getattr(self.ds_train, "wants_epoch", False), **bs)
        self.val_sampler = EvalBatchSampler(self.ds_val.lengths, seed=self.seed, max_batches=self.eval_max_batches, **bs)
        nw, vnw = int(dc["num_workers"]), int(dc["val_num_workers"])
        self.dl_train = DataLoader(self.ds_train, batch_sampler=self.sampler, collate_fn=self.ds_train.collate,
                                   num_workers=nw, pin_memory=bool(dc["pin_memory"]), persistent_workers=nw > 0)
        self.dl_val = DataLoader(self.ds_val, batch_sampler=self.val_sampler, collate_fn=self.ds_val.collate,
                                 num_workers=vnw, pin_memory=bool(dc["pin_memory"]), persistent_workers=vnw > 0)
        # 추가 검증 세트(선택, 훅이 ds_val_unseen 을 만들었을 때) — 처음 보는 화자 → TB val_unseen/*
        self.extra_vals = {}
        if getattr(self, "ds_val_unseen", None) is not None:
            vs = EvalBatchSampler(self.ds_val_unseen.lengths, seed=self.seed, max_batches=None, **bs)
            self.extra_vals["val_unseen"] = (vs, DataLoader(self.ds_val_unseen, batch_sampler=vs, collate_fn=self.ds_val_unseen.collate,
                                                            num_workers=0, pin_memory=bool(dc["pin_memory"])))
        for name, dsv in getattr(self, "ds_val_extra", {}).items():      # data.val_unseen_sets (2026-10-28) → TB <name>/*
            vs = EvalBatchSampler(dsv.lengths, seed=self.seed, max_batches=None, **bs)
            self.extra_vals[name] = (vs, DataLoader(dsv, batch_sampler=vs, collate_fn=dsv.collate, num_workers=0,
                                                    pin_memory=bool(dc["pin_memory"])))
        # 추가 학습 데이터: 소스마다 따로 도는 샘플러(원래 데이터의 순서·선택과 무관) — 훅이 self.extra_train 을 만들었을 때
        self.extra_srcs = []
        for i, (name, dse, per_step) in enumerate(getattr(self, "extra_train", [])):
            n_micro = int(per_step) // (int(dc["batch_size"]) * acc.num_processes)
            es = BucketBatchSampler(dse.lengths, seed=self.seed + 100_003 * (i + 1), multiple_of=n_micro,
                                    tag_epoch=getattr(dse, "wants_epoch", False), **bs)
            enw = max(1, nw // 2) if nw > 0 else 0
            dle = DataLoader(dse, batch_sampler=es, collate_fn=dse.collate, num_workers=enw,
                             pin_memory=bool(dc["pin_memory"]), persistent_workers=enw > 0)
            self.extra_srcs.append({"name": name, "ds": dse, "per_step": int(per_step), "n_micro": n_micro,
                                    "sampler": es, "loader": dle, "it": None})
        if sum(x["n_micro"] for x in self.extra_srcs) != self.extra_micro:
            _stop(f"❌ 추가 데이터 micro-batch 수 불일치: 설정 {self.extra_micro} · 데이터셋 "
                  f"{sum(x['n_micro'] for x in self.extra_srcs)} — build_datasets 가 data.extra_sources 를 모두 만들었는지 확인하세요.")
        self.steps_per_epoch = max(len(self.sampler) // self.grad_accum, 1)

        groups = {"decay": [], "no_decay": []}
        for n, p in self.module.named_parameters():
            if p.requires_grad:
                groups["no_decay" if any(k in n for k in _NO_DECAY) else "decay"].append(p)
        n_trainable = sum(p.numel() for g in groups.values() for p in g)
        if n_trainable == 0:                                  # 모든 rank 가 같은 결론 → 함께 중단
            _stop("❌ 학습할 파라미터가 0개입니다 — lora 설정 · freeze 를 확인하세요.")
        self.base_lr, self.wd = float(oc["lr"]), float(oc["weight_decay"])
        self.betas, self.eps = tuple(oc["betas"]), float(oc["eps"])
        opt = torch.optim.AdamW([{"params": groups["decay"], "weight_decay": self.wd, "name": "decay"},
                                 {"params": groups["no_decay"], "weight_decay": 0.0, "name": "no_decay"}],
                                lr=self.base_lr, betas=self.betas, eps=self.eps)
        # smoke 는 warmup 을 1 로 — 몇 step 안에 실제 lr 로 움직여야 "가중치가 학습되는지" 를 확인할 수 있다
        self.sched_cfg = {"name": oc["scheduler"], "warmup": 1 if smoke else int(oc["warmup_steps"]),
                          "max_steps": self.max_steps}
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda(self.sched_cfg["name"], self.sched_cfg["warmup"], self.max_steps))
        self.model, self.optimizer = acc.prepare(self.module, opt)   # DataLoader·scheduler 는 prepare 하지 않음
        self.state = ckpt.TrainState()
        acc.register_for_checkpointing(self.state)                   # custom_checkpoint_0 (순서 고정!)
        acc.register_for_checkpointing(self.scheduler)               # custom_checkpoint_1

        # ── ⑤ resume ──
        if self.mode == "resume":
            acc.load_state(str(self.policy["resume_from"]))
            self._reconcile_after_resume(allow_schedule_change)
            if acc.is_main_process:
                logger.info(c(f"↻ resume ← {self.policy['resume_from']}  (step {self.state.global_step} · "
                              f"epoch {self.state.epoch} · 배치 {self.state.batch_in_epoch} 부터 이어서)", "MAGENTA", "BRIGHT"))
        self.state.max_steps, self.state.sched, self.state.seed = self.max_steps, dict(self.sched_cfg), self.seed
        self._init_extra_state(allow_schedule_change)

        # ── ⑥ TensorBoard · 스냅샷 ──
        if self.use_tb:
            # config= 를 넘기지 않는다: add_hparams 가 {run}/{timestamp}/ 유령 run 을 만든다 → TEXT 탭에 표로
            acc.init_trackers(project_name=self.run_name)
            if acc.is_main_process and not acc.trackers:
                raise RuntimeError("TensorBoard 트래커 초기화 실패(tensorboard 미설치?) — accelerate 가 조용히 드롭했습니다.")
            if acc.is_main_process:
                self._tb_text("config/hparams", "| key | value |\n| --- | --- |\n" + "\n".join(
                    f"| {k} | {v} |" for k, v in flatten(cfg).items()), self.state.global_step)
        if acc.is_main_process and not smoke:
            snap = ckpt.write_snapshot(cfg, self.mode, acc.num_processes, self.seed)
            logger.info(f"설정 스냅샷 → {snap}")
        self._target_logged = False

    # ────────────────────────────────────────────── 훅 (서브클래스)
    def build_model(self):
        raise NotImplementedError

    def build_datasets(self):
        raise NotImplementedError

    def preflight(self):
        """학습 시작 전 검사(예: export 조건). 실패하면 예외 — 몇 시간 뒤 저장에서 터지지 않게."""

    def forward_step(self, model, batch: dict) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def loss_from_sums(self, out: Dict[str, torch.Tensor], denom: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def metrics_from_sums(self, sums: Dict[str, float]) -> Dict[str, float]:
        raise NotImplementedError

    def export(self, step_dir: Path) -> List[Tuple[str, Path]]:
        return []

    def verify_export(self, fmt: str, path: Path) -> None:
        pass

    def media_rows(self) -> List[dict]:
        return []

    def synthesize(self, rows: List[dict]) -> List[np.ndarray]:
        return []

    def banner_sections(self) -> list:
        return []

    def describe_data(self) -> List[str]:
        return []

    # ────────────────────────────────────────────── 공통 도우미
    def _tb_writer(self):
        return self.acc.get_tracker("tensorboard", unwrap=True) if self.use_tb else None

    def _tb_text(self, tag, text, step):
        try:
            self._tb_writer().add_text(tag, text, step)
        except Exception as e:  # noqa: BLE001 — 로깅 실패로 학습을 막지 않는다
            logger.warning("[tb] %s 기록 실패(학습은 계속): %s", tag, e)

    def _to_device(self, batch: dict) -> dict:
        dev = self.acc.device
        return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    @staticmethod
    def _mem_gb() -> float:
        return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0

    def _cur_lr(self) -> float:
        """실제로 이번 step 에 쓰일 lr — optimizer 에서 읽는다(get_last_lr 은 resume 직후 낡은 값일 수 있음)."""
        return float(self.optimizer.param_groups[0]["lr"])

    def _reconcile_after_resume(self, allow_change: bool):
        """resume 뒤 스케줄·optimizer 하이퍼를 현재 설정에 맞추고, 바뀐 것을 알린다."""
        st, warn = self.state, (logger.warning if self.acc.is_main_process else (lambda *a: None))
        saved = dict(st.sched or {})
        diff = {k: (saved.get(k), self.sched_cfg[k]) for k in self.sched_cfg if saved and saved.get(k) != self.sched_cfg[k]}
        if diff:
            msg = ("스케줄이 체크포인트와 다릅니다: " + ", ".join(f"{k} {a}→{b}" for k, (a, b) in diff.items())
                   + "\n   lr 곡선이 resume 지점에서 바뀝니다(cosine 이면 급격히 바뀔 수 있음).")
            if not allow_change:
                _stop("❌ " + msg + "\n   의도한 것이면 --allow_schedule_change 를 주세요.")
            warn("⚠️ " + msg)
        if st.seed is not None and st.seed != self.seed:
            warn(f"⚠️ seed 가 체크포인트({st.seed})와 다릅니다({self.seed}) — 데이터 순서가 달라집니다.")
        s = self.scheduler
        changed = []
        for g, lam in zip(self.optimizer.param_groups, s.lr_lambdas):
            want_wd = self.wd if g.get("name") == "decay" else 0.0
            for key, want in (("weight_decay", want_wd), ("betas", self.betas), ("eps", self.eps), ("initial_lr", self.base_lr)):
                have = tuple(g[key]) if key == "betas" else g.get(key)
                if have != want:
                    changed.append(f"{g.get('name')}.{key} {have}→{want}")
                    g[key] = want
            g["lr"] = self.base_lr * lam(s.last_epoch)
        s.base_lrs = [self.base_lr for _ in s.base_lrs]
        s._last_lr = [g["lr"] for g in self.optimizer.param_groups]
        if changed:
            warn("⚠️ optimizer 설정을 현재 yaml 값으로 맞췄습니다: " + ", ".join(changed))

    def _init_extra_state(self, allow_change: bool):
        """추가 데이터 스트림 위치(TrainState.extra)를 정한다: resume 이면 체크포인트 위치 그대로, 아니면 처음부터."""
        st, warn = self.state, (logger.warning if self.acc.is_main_process else (lambda *a: None))
        saved = dict(st.extra or {})
        st.extra = {}
        for x in self.extra_srcs:
            prev = saved.get(x["name"])
            if self.mode == "resume" and prev is None:
                warn(f"⚠️ 체크포인트에 추가 데이터 '{x['name']}' 위치가 없습니다 — 그 스트림은 처음부터 시작합니다.")
            elif prev is not None and int(prev.get("per_step", -1)) != x["per_step"]:
                msg = (f"추가 데이터 '{x['name']}' 의 per_step 이 체크포인트와 다릅니다: {prev.get('per_step')}→{x['per_step']}"
                       " — 그 스트림의 순서가 바뀝니다(그 epoch 의 처음부터).")
                if not allow_change:
                    _stop("❌ " + msg + "\n   의도한 것이면 --allow_schedule_change 를 주세요.")
                warn("⚠️ " + msg)
                prev = {"epoch": int(prev.get("epoch", 0)), "batch_in_epoch": 0}
            prev = prev or {"epoch": 0, "batch_in_epoch": 0}
            st.extra[x["name"]] = {"epoch": int(prev["epoch"]), "batch_in_epoch": int(prev["batch_in_epoch"]),
                                   "per_step": x["per_step"]}
        if saved and not self.extra_srcs:
            warn(f"⚠️ 체크포인트에는 추가 데이터({', '.join(saved)})가 있었지만 지금 설정에는 없습니다 — 원래 데이터만 학습합니다.")

    def _next_extra(self, x: dict) -> list:
        """추가 데이터 소스 x 의 다음 n_micro 개 micro-batch. epoch 이 끝나면 다음 epoch(새 순서)으로 넘어간다.
        위치는 TrainState.extra 에 바로 적는다 — 저장은 step 이 끝난 뒤에만 일어나므로 resume 이 정확하다."""
        es = self.state.extra[x["name"]]
        if x["it"] is None:
            x["sampler"].set_epoch(es["epoch"], start=es["batch_in_epoch"])
            x["it"] = iter(x["loader"])
        w = list(itertools.islice(x["it"], x["n_micro"]))
        if len(w) < x["n_micro"]:                                   # 샘플러가 n_micro 배수로 맞춰 둠 → 여기선 항상 0개
            es["epoch"], es["batch_in_epoch"] = es["epoch"] + 1, 0
            x["sampler"].set_epoch(es["epoch"])
            x["it"] = iter(x["loader"])
            w = list(itertools.islice(x["it"], x["n_micro"]))
            if len(w) < x["n_micro"]:
                raise RuntimeError(f"추가 데이터 '{x['name']}' 가 비어 있습니다(배치 {len(w)}/{x['n_micro']})")
        es["batch_in_epoch"] += x["n_micro"]
        return w

    # ────────────────────────────────────────────── 배너
    def _banner(self):
        if not self.acc.is_main_process:
            return
        dc, oc, tc = self.cfg["data"], self.cfg["optim"], self.cfg["train"]
        n = self.acc.num_processes
        bs = f"max_tokens={dc['max_tokens']}" if dc.get("max_tokens") else f"batch_size={dc['batch_size']}/GPU"
        mode = {"smoke": "SMOKE", "fresh": "새로 시작", "resume": f"이어서 (step {self.state.global_step})"}[self.mode]
        console.banner(f"{self.title} — {self.run_name}  [{mode}]", [
            ("Run", [("run_dir", self.run_dir), ("tensorboard", self.tb_dir if self.use_tb else "off"),
                     ("seed", f"{self.seed} (데이터 순서 = seed + epoch)")]),
            *self.banner_sections(),
            ("Data", [("train", f"{len(self.ds_train)} utt (제외 {getattr(self.ds_train, 'n_dropped', 0)})"),
                      *([("val_unseen", f"{len(self.ds_val_unseen)} utt · 처음 보는 화자 "
                          f"{len({r.get('speaker') for r in self.ds_val_unseen.rows})}명 (TB val_unseen/*)")]
                        if getattr(self, "ds_val_unseen", None) is not None else []),
                      *[(name, f"{len(dsv)} utt · 처음 보는 화자 {len({r.get('speaker') for r in dsv.rows})}명 (TB {name}/*)"
                               + (f" · 지시문 없는 문장 {dsv.n_no_instruct}" if getattr(dsv, "n_no_instruct", 0) else ""))
                        for name, dsv in getattr(self, "ds_val_extra", {}).items()],
                      *[(f"+ {x['name']}", f"{len(x['ds'])} utt · 한 step 에 {x['per_step']}문장 더함 "
                                           f"(GPU 마다 micro-batch {x['n_micro']}개 · 따로 도는 순서) · TB train_{x['name']}/*")
                        for x in self.extra_srcs],
                      ("val", f"{len(self.ds_val)} utt · " + ("val 전체" if self.eval_max_batches is None else
                                                              f"앞 {self.val_sampler.total} 배치만(⚠️ 비교 주의)")
                              + f" · 평가 배치 {self.val_sampler.total}"),
                      ("batch", f"{bs} · bucket ×{dc['bucket_mult']}"),
                      ("steps/epoch", f"{self.steps_per_epoch} (optimizer step 기준)")]),
            ("Optim", [("AdamW", f"lr={oc['lr']} wd={oc['weight_decay']} clip={self.grad_clip} (bias·norm 은 wd 0)"),
                       ("scheduler", f"{self.sched_cfg['name']} warmup={self.sched_cfg['warmup']}"),
                       ("loss 정규화", "창·전 GPU 의 target 프레임 수 합 (모든 프레임 같은 가중치)")]),
            ("Train", [("max_steps", f"{self.max_steps}  (≈ {self.max_steps / self.steps_per_epoch:.1f} epoch)"),
                       ("accumulate", f"{self.grad_accum} × {n} GPU → 유효 배치 ×{self.grad_accum * n}"
                        + (f" · + 추가 micro-batch {self.extra_micro}/GPU(창 {self.window}) → 한 step = 원래 "
                           f"{self.grad_accum * n * int(dc['batch_size'])} + 추가 {sum(x['per_step'] for x in self.extra_srcs)}문장"
                           if self.extra_srcs else "")),
                       ("every", f"log {self.log_every or '끔'} · eval {self.eval_every or '끔'} · "
                                 f"media {self.media_every or '끔'} · save {self.save_every or '끔'} (keep {self.keep_last})"),
                       ("resume", (f"재개용 파일 {self.resume_every or self.save_every} step 마다 · 최근 {self.resume_keep}개 "
                                   f"(export 는 save 마다)") if self.resume_enable else "끔(export 만 — 끊기면 처음부터)"),
                       ("early_traj", ", ".join(map(str, sorted(self.early_traj))) or "-"),
                       ("export", f"step_N/export_{getattr(self, 'export_kind', '-')}/ · dtype {getattr(self, 'export_dtype_note', None) or self.ec.get('dtype', '-')}")]),
        ])
        if self._eval_cap_warning:
            logger.warning(f"⚠️ train.eval_max_batches={self.eval_max_batches} — val 의 일부만 평가합니다. 배치 설정이 다른 실험과는 "
                           "val 을 비교할 수 없습니다(전체 평가: null).")
        lines = self.describe_data()
        if lines:
            logger.info(c("데이터 샘플(정규화문 확인):", "CYAN"))
            for ln in lines:
                logger.info(c("   · ", "CYAN") + ln)

    # ────────────────────────────────────────────── 학습 루프
    def fit(self):
        acc, st = self.acc, self.state
        self._banner()
        if acc.is_main_process:
            console.rule("Training Start", "GREEN")
        self.model.train()
        pbar = tqdm(total=self.max_steps, initial=st.global_step, disable=not acc.is_main_process,
                    desc="train", unit="step", dynamic_ncols=True)
        dev = acc.device
        win_sums = torch.zeros(len(self.sum_keys), device=dev)
        extra_sums = {x["name"]: torch.zeros(len(self.sum_keys), device=dev) for x in self.extra_srcs}
        t_last, step_last, t_start, step_start = time.time(), st.global_step, time.time(), st.global_step
        checked_grads = False
        try:
            with logging_redirect_tqdm():
                while st.global_step < self.max_steps:
                    self.sampler.set_epoch(st.epoch, start=st.batch_in_epoch)
                    it = iter(self.dl_train)
                    epoch_done = False
                    while st.global_step < self.max_steps:
                        window = list(itertools.islice(it, self.grad_accum))
                        if len(window) < self.grad_accum:          # epoch 끝 (샘플러가 창 배수로 맞춰 둠)
                            epoch_done = True
                            break
                        # 추가 데이터: 원래 창이 다 찬 뒤에만 꺼낸다(epoch 경계에서 낭비·어긋남 없음). 없으면 빈 목록
                        extra = [(x["name"], b) for x in self.extra_srcs for b in self._next_extra(x)]
                        batches = window + [b for _, b in extra]
                        tags = [None] * len(window) + [nm for nm, _ in extra]
                        # loss 정규화 = 창(원래 + 추가) · 전 GPU 의 정답 token 수 합 → 모든 token 같은 가중치
                        denom = acc.reduce(torch.stack([b["counts"] for b in batches]).sum(0).to(dev), "sum")
                        gn = None
                        for b, tag in zip(batches, tags):
                            b = self._to_device(b)
                            with acc.accumulate(self.model):
                                out = self.forward_step(self.model, b)
                                # accelerate 가 /window(= grad_accum + 추가), DDP 가 /world 를 하므로 되돌려 곱한다
                                loss = self.loss_from_sums(out, denom) * (self.window * acc.num_processes)
                                acc.backward(loss)
                                if acc.sync_gradients:
                                    if self.smoke and not checked_grads:
                                        self._check_grads()
                                        checked_grads = True
                                    gn = acc.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                                self.optimizer.step()
                                self.optimizer.zero_grad(set_to_none=True)
                            # train/* = 원래 데이터만(이전 run 과 곡선 비교 가능) · train_<name>/* = 추가 데이터
                            (win_sums if tag is None else extra_sums[tag]).add_(
                                torch.stack([out[k].detach().float() for k in self.sum_keys]))
                        lr = self._cur_lr()                         # scheduler.step() 전 = 이번 step 의 lr
                        self.scheduler.step()
                        st.global_step += 1
                        st.batch_in_epoch += self.grad_accum
                        gstep = st.global_step
                        pbar.update(1)

                        if (self.log_every and (gstep % self.log_every == 0 or gstep == step_start + 1)) or self.smoke:
                            sums = acc.reduce(win_sums, "sum").tolist()
                            m = self.metrics_from_sums(dict(zip(self.sum_keys, sums)))
                            n_steps, dt = gstep - step_last, time.time() - t_last
                            tps = sums[self.sum_keys.index("t_n")] / max(dt, 1e-9) if "t_n" in self.sum_keys else None
                            m.update(lr=lr, grad_norm=float(gn) if gn is not None else None, epoch=st.epoch)
                            eta = (time.time() - t_start) / max(gstep - step_start, 1) * (self.max_steps - gstep)
                            if acc.is_main_process:
                                logger.info(f"{c(f'step {gstep:>6}/{self.max_steps}', 'BRIGHT')} | "
                                            + fmt_metrics(m, ["loss", "loss_diff", "loss_stop", "lr", "grad_norm"])
                                            + f" | {dt / max(n_steps, 1):.2f}s/step"
                                            + (f" · {tps:,.0f} frame/s" if tps else "")
                                            + f" | mem {self._mem_gb():.1f}G | ETA {fmt_seconds(eta)}")
                                pbar.set_postfix(loss=f"{m['loss']:.3f}", lr=f"{lr:.1e}")
                            acc.log({f"train/{k}": v for k, v in m.items() if v is not None}
                                    | {"mem/max_allocated_gb": self._mem_gb(),
                                       "speed/s_per_step": dt / max(n_steps, 1)}
                                    | ({"speed/tokens_per_s": tps} if tps else {}), step=gstep)
                            win_sums.zero_()
                            for nm, es_ in extra_sums.items():
                                em = self.metrics_from_sums(dict(zip(self.sum_keys, acc.reduce(es_, "sum").tolist())))
                                if acc.is_main_process:
                                    logger.info(f"{'':>13}+ {nm} | " + fmt_metrics(em, ["loss", "loss_diff", "loss_stop"]))
                                acc.log({f"train_{nm}/{k}": v for k, v in em.items() if v is not None}, step=gstep)
                                es_.zero_()
                            t_last, step_last = time.time(), gstep
                        if not self.smoke:
                            if self.eval_every and gstep % self.eval_every == 0:
                                self.validate(gstep)
                            if self.media_every and gstep % self.media_every == 0:
                                self.log_media(gstep)
                            if self.save_every and gstep % self.save_every == 0:
                                self.save(gstep)
                            elif gstep in self.early_traj:
                                self.save(gstep)
                    if epoch_done:                                   # 다음 epoch 은 처음부터(새 순서)
                        st.epoch += 1
                        st.batch_in_epoch = 0
        except BaseException:
            if self.use_tb and acc.is_main_process:
                try:
                    self._tb_writer().flush()
                except Exception:  # noqa: BLE001
                    pass
            raise
        finally:
            pbar.close()

        if self.smoke:
            self.validate(st.global_step)
            self._smoke_checks()
            if acc.is_main_process:
                console.rule(f"SMOKE OK — {self.max_steps} step · peak mem {self._mem_gb():.1f} GB", "GREEN")
        else:
            if not (self.save_every and st.global_step % self.save_every == 0):
                self.save(st.global_step)
            if acc.is_main_process:
                console.rule(f"Training End — step {st.global_step} · {fmt_seconds(time.time() - t_start)}", "GREEN")
        acc.end_training()

    def _check_grads(self):
        """smoke: 학습 대상인데 gradient 가 없는 파라미터(= forward 에서 안 쓰임 → DDP 오류 원인) 검사."""
        mod = self.acc.unwrap_model(self.model)
        dead = [n for n, p in mod.named_parameters() if p.requires_grad and p.grad is None]
        if dead:
            _stop(f"❌ gradient 가 없는 학습 대상 파라미터 {len(dead)}개 (예: {dead[:5]}) — "
                             "freeze 하거나 train.ddp_find_unused: true")
        if self.acc.is_main_process:
            logger.info(c("✓ [smoke] 모든 학습 대상 파라미터에 gradient 가 흐릅니다.", "GREEN"))

    # ────────────────────────────────────────────── validation
    @torch.no_grad()
    def _eval_loader(self, model, loader) -> Tuple[dict, dict]:
        """한 검증 로더의 합계 → (지표, 합계). DDP 를 거치지 않는 forward + 합계 all-reduce."""
        acc = self.acc
        sums = torch.zeros(len(self.sum_keys), device=acc.device)
        with preserved_rng(self.cfg["train"].get("eval_seed")):
            for batch in loader:
                batch = self._to_device(batch)
                with acc.autocast():
                    out = self.forward_step(model, batch)
                sums += torch.stack([out[k].detach().float() for k in self.sum_keys])
        sums = dict(zip(self.sum_keys, acc.reduce(sums, "sum").tolist()))
        return self.metrics_from_sums(sums), sums

    @torch.no_grad()
    def validate(self, step: int) -> dict:
        """DDP 를 거치지 않는 forward(collective 없음 → rank 별 배치 수가 달라도 안전) + 합계 all-reduce.
        추가 검증 세트(extra_vals, 예: 처음 보는 화자)가 있으면 같은 방식으로 이어서 평가해 <이름>/* 로 기록한다."""
        acc = self.acc
        model = acc.unwrap_model(self.model)
        model.eval()
        keys = ["loss", "loss_diff", "loss_stop"]
        m, n = self._eval_loader(model, self.dl_val)
        if acc.is_main_process:
            tqdm.write("")
            logger.info(c(f"🔎 [val] step {step} | " + fmt_metrics(m, keys)
                          + f" | {int(n.get('t_n', 0))} frame · 배치 {self.val_sampler.total}", "YELLOW", "BRIGHT"))
        acc.log({f"val/{k}": v for k, v in m.items()}, step=step)
        for name, (vs, loader) in getattr(self, "extra_vals", {}).items():
            em, en = self._eval_loader(model, loader)
            if acc.is_main_process:
                logger.info(c(f"🔎 [{name}] step {step} | " + fmt_metrics(em, keys)
                              + f" | {int(en.get('t_n', 0))} frame · 배치 {vs.total}", "CYAN", "BRIGHT"))
            acc.log({f"{name}/{k}": v for k, v in em.items()}, step=step)
        if acc.is_main_process:
            tqdm.write("")
        model.train()
        return m

    # ────────────────────────────────────────────── media (TensorBoard)
    def log_media(self, step: int):
        acc = self.acc
        if acc.is_main_process and self.use_tb:
            t0, mem0 = time.time(), torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0
            writer, rows, n_ok = self._tb_writer(), self.media_rows(), 0

            def safe(fn, tag, *a, **kw):
                try:
                    fn(tag, *a, **kw)
                    return True
                except Exception as e:  # noqa: BLE001
                    logger.warning("[tb write] %s 실패(학습은 계속): %s", tag, e)
                    import matplotlib.pyplot as plt
                    plt.close("all")
                    return False

            try:
                with preserved_rng(int(self.mc.get("seed", self.seed))):
                    wavs = self.synthesize(rows)
                for i, (r, w) in enumerate(zip(rows, wavs)):
                    cap = f"{r['utt']} | {r.get('speaker')} | {r.get('emotion')} | {r['text']}"
                    if "ref_utt" in r:                        # 2026-11-08: REF 를 캡션에 · REF 소리도 처음 한 번
                        cap += (f" | REF {r['ref_utt']} ({r.get('ref_speaker')} · {r.get('ref_emotion')}) | {r.get('src')}"
                                if r.get("ref") else " | REF 없음(목소리 비교 금지)")
                    if not self._target_logged and r.get("ref"):
                        rw = load_audio(r["ref"], SR_IN)
                        safe(writer.add_audio, f"media/ref_audio_{i}", np.clip(rw, -1, 1), step, sample_rate=SR_IN)
                        safe(writer.add_text, f"media/ref_caption_{i}", f"[REF] {r['ref_utt']} | {r.get('ref_speaker')} | {r.get('ref_emotion')}", step)
                    if not self._target_logged:
                        gt = load_audio(r["audio"], SR_IN)
                        safe(writer.add_audio, f"media/target_audio_{i}", np.clip(gt, -1, 1), step, sample_rate=SR_IN)
                        safe(writer.add_figure, f"media/target_mel_{i}", mel_to_figure(mel(gt, SR_IN), f"[GT] {r['utt']}"), step, close=True)
                        safe(writer.add_text, f"media/target_caption_{i}", "[GT] " + cap, step)
                    safe(writer.add_audio, f"media/pred_audio_{i}", np.clip(w, -1, 1), step, sample_rate=SR_OUT)
                    safe(writer.add_figure, f"media/pred_mel_{i}", mel_to_figure(mel(w, SR_OUT), f"[pred@{step}] {r['utt']}"), step, close=True)
                    safe(writer.add_text, f"media/pred_caption_{i}", f"[pred@{step}, {len(w) / SR_OUT:.2f}s] " + cap, step)
                    n_ok += 1
                self._target_logged = True
                writer.flush()
                mem1 = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0
                logger.info(c(f"🎵 [media] step {step} | {n_ok}/{len(rows)} 샘플 → TensorBoard ({time.time() - t0:.1f}s · "
                              f"mem {mem0:.1f}→{mem1:.1f}G, peak {self._mem_gb():.1f}G)", "BLUE", "BRIGHT"))
            except Exception as e:  # noqa: BLE001 — media 실패로 학습을 멈추지 않는다
                logger.warning("[media] step %d 실패(학습은 계속): %s", step, e)
        acc.wait_for_everyone()

    # ────────────────────────────────────────────── save / export
    def _resume_due(self, step: int) -> bool:
        """이 step 에 재개용 파일(save_state)을 저장하나 — resume.every 경계를 새로 넘은 저장 step(배수면 그 step)이거나 마지막 step."""
        if not self.resume_enable:
            return False
        if step >= self.max_steps or not self.resume_every:
            return True
        last = ckpt.latest_resumable(self.run_dir)       # every 가 save_every 의 배수가 아니어도: 경계를 넘은 첫 저장 step
        return step // self.resume_every > (ckpt.step_of(last) // self.resume_every if last else 0)

    def save(self, step: int):
        acc = self.acc
        save_dir = self.run_dir / f"step_{step}"
        if acc.is_main_process:
            ckpt.rotate(self.run_dir, self.keep_last - 1, protect=save_dir)  # 저장 전에 정리 → 디스크 순간 최대치 감소
        acc.wait_for_everyone()
        with_state = self._resume_due(step)
        if with_state:
            acc.save_state(str(save_dir))
        elif acc.is_main_process:
            save_dir.mkdir(parents=True, exist_ok=True)
        if acc.is_main_process:
            t0 = time.time()
            try:
                made = self.export(save_dir)
                for fmt, path in made:
                    self.verify_export(fmt, path)
                ckpt.mark_complete(save_dir, step=step, epoch=self.state.epoch, exports=[f for f, _ in made], resume=with_state)
                if with_state:
                    for p in ckpt.prune_resume(self.run_dir, self.resume_keep, protect=save_dir):
                        logger.info(c(f"🧹 [resume] {p.name} 의 재개용 파일 삭제(최근 {self.resume_keep}개만 보관 · export 는 그대로)", "MAGENTA"))
                if step in self.early_traj:
                    dst = Path(self.cfg["_derived"]["save_root"]) / "_early_traj" / self.run_name / save_dir.name
                    for fmt, path in made:
                        ckpt.link_or_copy(path, dst / path.name)
                    logger.info(c(f"📌 [early_traj] step {step} export 보관 → {dst} (회전과 무관)", "MAGENTA"))
                logger.info(c(f"💾 [save] step {step} → {save_dir}  (export: {', '.join(f for f, _ in made) or '-'}, "
                              f"재개용 파일 {'저장' if with_state else '없음(export 만)'}, "
                              f"keep_last={self.keep_last}, {time.time() - t0:.1f}s)", "MAGENTA", "BRIGHT"))
            except Exception as e:  # noqa: BLE001 — 학습 상태는 저장됨. 결과물이 없다는 사실을 크게 남긴다
                (save_dir / ckpt.EXPORT_FAILED_MARK).write_text(traceback.format_exc())
                logger.error(c(f"❌ [save] step {step} export 실패 — 학습 상태(save_state)는 저장됨(resume 가능). "
                               f"{save_dir / ckpt.EXPORT_FAILED_MARK} 참고: {e}", "RED", "BRIGHT"))
                self._tb_text("errors/export", f"step {step}: {e}", step)
            ckpt.rotate(self.run_dir, self.keep_last, protect=save_dir)
        acc.wait_for_everyone()

    def _smoke_checks(self):
        """smoke: 합성 1개(wav 저장) + export 를 임시 폴더로 1회 실행·검증(그 뒤 삭제)."""
        acc = self.acc
        if acc.is_main_process:
            out = self.run_dir / "_smoke"
            out.mkdir(parents=True, exist_ok=True)
            rows = self.media_rows()                     # 2026-11-08: media 문장 전부(REF 와 같은 목소리인지 smoke 에서 확인)
            if rows and self.mc.get("enable", True):
                with preserved_rng(int(self.mc.get("seed", self.seed))):
                    ws = self.synthesize(rows)
                for i, (r, w) in enumerate(zip(rows, ws)):
                    sf.write(out / f"media{i}_{r['utt']}.wav", w, SR_OUT)
                    if r.get("ref"):
                        shutil.copy2(r["ref"], out / f"media{i}_REF_{r['ref_utt']}.wav")
                (out / "media_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
                logger.info(c(f"🎵 [smoke] media 합성 {len(ws)}개 OK ({' · '.join(f'{len(w) / SR_OUT:.2f}s' for w in ws)}) "
                              f"· REF 있음 {sum(1 for r in rows if r.get('ref'))}/{len(rows)} → {out}", "BLUE", "BRIGHT"))
            tmp = out / "export_check"
            made = self.export(tmp)
            for fmt, path in made:
                self.verify_export(fmt, path)
            shutil.rmtree(tmp, ignore_errors=True)
            if made:
                logger.info(c(f"💾 [smoke] export {', '.join(f for f, _ in made)} 생성·검증 OK (임시 폴더는 삭제)", "MAGENTA", "BRIGHT"))
        acc.wait_for_everyone()
