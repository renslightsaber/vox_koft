"""VoxCPM2 LoRA 트레이너 — 전체 FT 트레이너(VoxSFTTrainer)를 상속. 다른 것은 학습 파라미터와 저장물뿐.

  · 주입: from_local(training=True, lora_config=LoRAConfig(V2)) → LoRA 이외 파라미터는 모두 freeze (공식과 같음)
  · export: step_N/export_lora/ = lora_weights.safetensors(키에 lora_ 포함, fp32) + lora_config.json
            {"base_model": <원본 경로>, "lora_config": {...}} — 공식 test_voxcpm_lora_infer.py · voxko.engine 이 그대로 읽는다.
  · 🔴 공식 추론 스크립트는 V1 LoRAConfig 를 import 한다. 우리는 V2(voxcpm.model.voxcpm2.LoRAConfig)를 쓴다.
"""
from __future__ import annotations

from voxko.common import console  # noqa: F401
from voxko.common.console import logger

import json
import shutil
from pathlib import Path
from typing import List, Tuple

import torch

from voxko.sft.trainer import VoxSFTTrainer

LORA_KEYS = ("enable_lm", "enable_dit", "enable_proj", "r", "alpha", "dropout")


class VoxLoRATrainer(VoxSFTTrainer):
    title = "VoxCPM2 LoRA FT"
    export_kind = "lora"
    export_dtype_note = "float32 (LoRA 가중치만)"

    def lora_config(self):
        from voxcpm.model.voxcpm2 import LoRAConfig
        lc = self.cfg["lora"]
        return LoRAConfig(**{k: lc[k] for k in LORA_KEYS})

    def preflight(self):
        super().preflight()
        if not self.cfg["lora"].get("enable"):
            raise ValueError("LoRA 트레이너인데 lora.enable=false 입니다 — 실험 yaml 에 lora.enable: true")
        model = self.module
        bad = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
        if bad:
            raise RuntimeError(f"LoRA 가 아닌 학습 파라미터 {len(bad)}개(예: {bad[:3]})")
        if not any("lora_" in n for n, _ in model.named_parameters()):
            raise RuntimeError("LoRA 파라미터가 없습니다 — lora.enable_lm/dit/proj 를 확인하세요")

    def export(self, step_dir: Path) -> List[Tuple[str, Path]]:
        from safetensors.torch import save_file
        out = self._export_dir(step_dir)
        tmp = out.with_name(out.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        model = self.acc.unwrap_model(self.model)
        sd = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items() if "lora_" in k}
        save_file(sd, str(tmp / "lora_weights.safetensors"))
        lc = {k: self.cfg["lora"][k] for k in LORA_KEYS}
        (tmp / "lora_config.json").write_text(json.dumps(
            {"base_model": self.cfg["_derived"]["init_model_dir"], "lora_config": lc,
             "voxko": {"run": self.run_name, "step": self.state.global_step, "LoRAConfig": "V2"}},
            ensure_ascii=False, indent=2))
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
        return [("lora", out)]

    def verify_export(self, fmt: str, path: Path) -> None:
        from safetensors import safe_open
        for n in ("lora_weights.safetensors", "lora_config.json"):
            if not (path / n).is_file():
                raise RuntimeError(f"LoRA export 에 {n} 가 없습니다 — {path}")
        with safe_open(str(path / "lora_weights.safetensors"), "pt") as f:
            keys = set(f.keys())
        want = {k for k in self.acc.unwrap_model(self.model).state_dict() if "lora_" in k}
        if not keys or keys != want:
            raise RuntimeError(f"LoRA export 키 불일치: 저장 {len(keys)} · 모델 {len(want)}")
        cfg = json.loads((path / "lora_config.json").read_text())
        if not Path(cfg["base_model"]).is_dir():
            raise RuntimeError(f"lora_config.json base_model 이 없습니다: {cfg['base_model']}")
