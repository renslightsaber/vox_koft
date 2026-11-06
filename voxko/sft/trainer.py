"""VoxCPM2 전체 FT 트레이너 (BaseTrainer 훅 구현). LoRA 는 voxko/lora/trainer.py 가 이것을 상속한다.

공식 부품을 import 해서 쓴다(🔒 VoxCPM/ 은 수정하지 않음)
  · 모델      voxcpm.model.voxcpm2.VoxCPM2Model.from_local(training=True) — fp32 master, accelerate autocast bf16
  · 배치 구성  voxcpm.training.packers.AudioFeatureProcessingPacker — VAE encode(16kHz, fp32) + [ref] text 101 target 102
  · loss      모델 forward 의 loss/diff(CFM) · loss/stop(CE). 둘 다 target 칸 평균 → ×칸 수 로 합계를 만들어
              창 · 전 GPU 의 칸 수로 다시 나눈다(BaseTrainer 규칙: 모든 칸 같은 가중치).
  · export    step_N/export_ft/ = 공식 추론이 그대로 여는 폴더(config.json · model.safetensors · audiovae · tokenizer)

공식 학습 스크립트와 다른 점(docs/reference/voxcpm2_training_code_walkthrough.md §5)
  · HF accelerate · 데이터 위치까지 정확한 resume · 스케줄 가드 · val 같은 표본 · smoke · _COMPLETE 검증
  · VAE 는 모델에서 떼어 DDP 밖에 둔다(공식도 같음). 원본 가중치 로드 때 빠진 키를 검사한다(공식은 strict=False 로 조용히 넘어감).
"""
from __future__ import annotations

from voxko.common import console  # noqa: F401  ← 가장 먼저(colorama → logging → tqdm)
from voxko.common.console import c, logger

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from voxko.common.trainer_base import BaseTrainer
from voxko.data.dataset import load_instruct_map, make_dataset, read_jsonl

EXPORT_COPY = ("config.json", "audiovae.pth", "audiovae.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "tokenizer.model", "generation_config.json")


@contextmanager
def _vae_attached(model, vae):
    """합성(generate)은 model.audio_vae 를 쓴다 → 잠깐 붙였다가 뗀다(DDP 파라미터 목록과 무관)."""
    model.audio_vae = vae
    try:
        yield model
    finally:
        del model.audio_vae


class VoxSFTTrainer(BaseTrainer):
    title = "VoxCPM2 전체 FT"
    sum_keys = ("diff_sum", "stop_sum", "t_n", "n_ref", "n_utt")
    export_kind = "ft"

    # ────────────────────────────── 모델
    def lora_config(self):
        return None

    def build_model(self):
        from voxcpm.model.voxcpm2 import VoxCPM2Model
        from voxcpm.training.packers import AudioFeatureProcessingPacker
        d = self.cfg["_derived"]
        path = d["init_model_dir"]
        arch = json.loads(Path(path, "config.json").read_text()).get("architecture", "voxcpm2")
        if str(arch).lower() not in ("voxcpm2",):
            raise ValueError(f"VoxCPM2 체크포인트가 아닙니다(architecture={arch}): {path}")
        model = VoxCPM2Model.from_local(path, optimize=False, training=True, device=str(self.acc.device),
                                        lora_config=self.lora_config())
        self._check_loaded(model, path)
        self.vae = model.audio_vae.to(self.acc.device).eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        del model.audio_vae                               # DDP 밖(학습 대상 아님) — 공식 스크립트와 같음
        self.out_sr = int(getattr(self.vae, "out_sample_rate", 48000))
        if int(self.vae.sample_rate) != 16000:
            raise ValueError(f"AudioVAE 입력 샘플레이트가 16000 이 아닙니다: {self.vae.sample_rate}")
        self.packer = AudioFeatureProcessingPacker(dataset_cnt=1, max_len=model.config.max_length,
                                                   patch_size=model.config.patch_size, feat_dim=model.config.feat_dim,
                                                   audio_vae=self.vae)
        self.max_len_model = int(model.config.max_length)
        self.tokenize = model.text_tokenizer
        lam = self.cfg["train"]["lambdas"]
        self.lam_diff, self.lam_stop = float(lam["diff"]), float(lam["stop"])
        n_all = sum(p.numel() for p in model.parameters())
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._param_info = (n_all, n_tr)
        return model

    def _check_loaded(self, model, path: str):
        """from_local 은 strict=False 로 불러 빠진 키를 말하지 않는다 → 키 이름으로 직접 확인."""
        from safetensors import safe_open
        f = Path(path, "model.safetensors")
        if not f.is_file():
            logger.warning(f"⚠️ {f} 가 없어 키 검사를 건너뜁니다")
            return
        with safe_open(str(f), "pt") as sf_:
            have = set(sf_.keys())
        want = {k for k in model.state_dict() if not k.startswith("audio_vae.") and "lora_" not in k}
        missing, unexpected = sorted(want - have), sorted(have - want)
        if missing:
            raise RuntimeError(f"체크포인트에 없는 모델 키 {len(missing)}개(예: {missing[:5]}) — {f}")
        if unexpected and self.acc.is_main_process:
            logger.warning(f"⚠️ 모델이 쓰지 않는 체크포인트 키 {len(unexpected)}개(예: {unexpected[:3]})")

    # ────────────────────────────── 데이터
    def build_datasets(self):
        d, dc, seed = self.cfg["_derived"], self.cfg["data"], self.seed
        self.instruct_map = load_instruct_map(d["emotion_map"])
        mk = lambda jsonl, pool, name, strict=True: make_dataset(jsonl, pool, self.tokenize, self.instruct_map, dc, seed, name,  # noqa: E731
                                                                 strict=strict)
        train = mk(d["train_jsonl"], d["ref_pool_jsonl"], "train")
        val = mk(d["val_jsonl"], d["ref_pool_jsonl"], "val", strict=False)        # 평가 세트: drop_only 비율이 안 맞으면 경고만
        if d.get("val_unseen_jsonl"):
            self.ds_val_unseen = mk(d["val_unseen_jsonl"], d["val_unseen_pool_jsonl"], "val_unseen", strict=False)
        if self.acc.is_main_process:
            for ds_ in (train, val, getattr(self, "ds_val_unseen", None)):
                if ds_ is not None and ds_.drop_warning:
                    logger.warning("⚠️ " + ds_.drop_warning)
        self.extra_train = []
        for e in d.get("extra_sources") or []:
            pool = str(Path(e["train_jsonl"]).with_name("ref_pool.jsonl"))
            ds = mk(e["train_jsonl"], pool if os.path.isfile(pool) else None, e["name"])
            self.extra_train.append((e["name"], ds, int(e["per_step"])))
        if len(train) == 0 or len(val) == 0:
            raise ValueError(f"데이터가 비었습니다: train {len(train)} · val {len(val)} (max_seq_len 을 확인)")
        return train, val

    def preflight(self):
        if self.max_len_model < int(self.cfg["data"]["max_seq_len"]):
            raise ValueError(f"data.max_seq_len({self.cfg['data']['max_seq_len']}) > 모델 max_length({self.max_len_model})")
        if self.cfg["lora"].get("enable") and self.export_kind == "ft":
            raise ValueError("lora.enable=true 인데 전체 FT 트레이너입니다 — `run.sh train lora` 를 쓰세요")

    # ────────────────────────────── 한 step
    def forward_step(self, model, batch: dict) -> Dict[str, torch.Tensor]:
        B = batch["text_tokens"].size(0)
        with torch.autocast("cuda", enabled=False):           # VAE encode 는 fp32 (공식과 같음)
            packed = self.packer(audio_tokens=batch["audio_tokens"], text_tokens=batch["text_tokens"],
                                 task_ids=torch.ones(B, dtype=torch.int32, device=self.acc.device),
                                 dataset_ids=torch.zeros(B, dtype=torch.int32, device=self.acc.device),
                                 is_prompts=[False] * B, ref_audio_tokens=batch.get("ref_audio_tokens"))
        progress = min(1.0, self.state.global_step / max(self.max_steps, 1))
        out = model(packed["text_tokens"], packed["text_mask"], packed["audio_feats"], packed["audio_mask"],
                    packed["loss_mask"], packed["position_ids"], packed["labels"], progress=progress)
        n = packed["loss_mask"].sum().float()
        if self.smoke or self.state.global_step == 0:      # 정규화 분모(collate 의 counts)가 packer 의 loss 칸 수와 같은지
            if int(n.item()) != int(batch["counts"].item()):
                raise RuntimeError(f"target 칸 수 불일치: packer {int(n.item())} · collate {int(batch['counts'].item())}")
        return {"diff_sum": out["loss/diff"].float() * n, "stop_sum": out["loss/stop"].float() * n, "t_n": n,
                "n_ref": batch["n_ref"].to(n.device), "n_utt": torch.tensor(float(B), device=n.device)}

    def loss_from_sums(self, out, denom):
        # denom = [diff_sum, stop_sum, t_n, n_ref, n_utt] 의 창·전 GPU 합 중 t_n(정답 칸 수)
        t = denom[self.sum_keys.index("t_n")] if denom.ndim else denom
        return (self.lam_diff * out["diff_sum"] + self.lam_stop * out["stop_sum"]) / t.clamp(min=1.0)

    def metrics_from_sums(self, s):
        n = max(s["t_n"], 1.0)
        m = {"loss_diff": s["diff_sum"] / n, "loss_stop": s["stop_sum"] / n}
        m["loss"] = self.lam_diff * m["loss_diff"] + self.lam_stop * m["loss_stop"]
        m["ref_frac"] = s["n_ref"] / max(s["n_utt"], 1.0)
        return m

    # ────────────────────────────── export
    def _export_dir(self, step_dir: Path) -> Path:
        return Path(step_dir) / f"export_{self.export_kind}"

    def export(self, step_dir: Path) -> List[Tuple[str, Path]]:
        from safetensors.torch import save_file
        out = self._export_dir(step_dir)
        tmp = out.with_name(out.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        model = self.acc.unwrap_model(self.model)
        dt = torch.bfloat16 if self.cfg["export"]["dtype"] == "bfloat16" else torch.float32
        sd = {k: v.detach().to("cpu", dt if v.is_floating_point() else v.dtype).contiguous()
              for k, v in model.state_dict().items()}
        save_file(sd, str(tmp / "model.safetensors"))
        src = Path(self.cfg["_derived"]["init_model_dir"])
        for name in EXPORT_COPY:
            if (src / name).is_file():
                shutil.copy2(src / name, tmp / name)
        (tmp / "voxko_export.json").write_text(json.dumps(
            {"kind": "ft", "run": self.run_name, "step": self.state.global_step, "base": str(src), "dtype": str(dt)},
            ensure_ascii=False, indent=1))
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
        return [("ft", out)]

    def verify_export(self, fmt: str, path: Path) -> None:
        from safetensors import safe_open
        need = ["config.json", "model.safetensors"]
        miss = [n for n in need if not (path / n).is_file()]
        if miss or not any((path / n).is_file() for n in ("audiovae.pth", "audiovae.safetensors")):
            raise RuntimeError(f"export 에 파일이 없습니다: {miss or 'audiovae'} — {path}")
        with safe_open(str(path / "model.safetensors"), "pt") as f:
            keys = set(f.keys())
        want = {k for k in self.acc.unwrap_model(self.model).state_dict()}
        if keys != want:
            raise RuntimeError(f"export 키 불일치: 빠짐 {len(want - keys)} · 남음 {len(keys - want)}")

    # ────────────────────────────── 합성(media · smoke)
    def media_rows(self) -> List[dict]:
        n = int(self.mc.get("num_samples", 2))
        ds = self.ds_val
        rows = []
        for i in range(min(n, len(ds))):
            p = ds.plan(i, 0)
            r = ds.rows[i]
            rows.append({"utt": r["utt"], "audio": r["audio"], "speaker": r.get("speaker"), "emotion": r.get("emotion"),
                         "text": p["text"], "ref": p["ref"]["audio"] if p["ref"] else None})
        return rows

    @torch.no_grad()
    def synthesize(self, rows: List[dict]) -> List[np.ndarray]:
        model = self.acc.unwrap_model(self.model)
        was = model.training
        model.eval()
        wavs = []
        try:
            with _vae_attached(model, self.vae), torch.autocast("cuda", dtype=torch.bfloat16):
                for r in rows:
                    w = model.generate(target_text=r["text"], reference_wav_path=r.get("ref") or "",
                                       inference_timesteps=int(self.mc.get("inference_timesteps", 10)),
                                       cfg_value=float(self.mc.get("cfg_value", 2.0)),
                                       max_len=int(self.mc.get("max_len", 1000)), seed=int(self.mc.get("seed", self.seed)))
                    wavs.append(w.squeeze().float().cpu().numpy())
        finally:
            model.train(was)
        return wavs

    # ────────────────────────────── 배너
    def banner_sections(self) -> list:
        n_all, n_tr = self._param_info
        dc = self.cfg["data"]
        return [("Model", [("init", self.cfg["_derived"]["init_model_dir"]),
                           ("params", f"전체 {n_all / 1e9:.2f}B · 학습 {n_tr / 1e6:.1f}M ({100 * n_tr / max(n_all, 1):.2f}%)"),
                           ("loss", f"{self.lam_diff}·diff + {self.lam_stop}·stop"),
                           ("sample rate", f"입력 16000 · 출력 {self.out_sr}")]),
                ("Cond", [("지시문", f"{Path(self.cfg['_derived']['emotion_map']).name} · {self.ds_train.instr_summary()}"),
                          ("ref", f"비율 {dc['ref']['ratio']} · 풀 {dc['ref']['pool']} {dc['ref']['min_dur']}~{dc['ref']['max_dur']}s · "
                                  f"풀 없는 문장 {self.ds_train.n_no_pool} · 길이로 제외 {self.ds_train.n_dropped}")])]

    def describe_data(self) -> List[str]:
        ds, lines = self.ds_train, []
        for i in range(min(3, len(ds))):
            p = ds.plan(i, 0)
            lines.append(f"{ds.rows[i]['utt']} | {ds.rows[i]['emotion']} | ref={p['ref']['utt'] if p['ref'] else '-'} | {p['text']}")
        return lines
