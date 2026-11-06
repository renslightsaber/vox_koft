"""VoxCPM2 합성 엔진 — 원본 · 전체 FT(export_ft) · LoRA(export_lora) 를 같은 방식으로 불러 합성.

    python -m voxko.engine.synth --model ~/workspaces/ourckpts/voxcpm/VoxCPM2 \
        --text "오늘 정말 기뻐요." --emotion Happy-B --ref <무감정 16k/24k wav> --out out.wav
    python -m voxko.engine.synth --model <run>/step_500/export_ft   ...       # 전체 FT
    python -m voxko.engine.synth --model <run>/step_500/export_lora ...       # LoRA (base 는 lora_config.json 에서)

규칙 (CLAUDE.md §4)
  · 한국어는 normalize=False + ko_tts_norm.normalize(t, profile="qwen3tts") — 내장 정규화는 숫자를 영어로 읽음.
  · 같은 seed 라도 모델 로드 뒤 **첫 합성만** 값이 다름 → load 뒤 준비 합성 1회(warmup=True 기본).
  · 순수 PyTorch(optimize=False) — sm_120 + LoRA + nano-vllm 은 소리가 나빠짐(공식 이슈 #326).
  · LoRA 는 V2 LoRAConfig 로 만든다(공식 core 의 자동 로드는 V1 클래스).
  · 지시문은 학습과 같은 표(configs/emotion/<표>.yaml)로 "(지시문)문장" 을 만든다. 모드 = Controllable cloning(reference + 지시문).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_BASE = os.path.expanduser("~/workspaces/ourckpts/voxcpm/VoxCPM2")
DEFAULT_MAP = str(Path(__file__).resolve().parents[2] / "configs/emotion/a7n_vox_v1.yaml")


def model_kind(path: str) -> str:
    p = Path(path)
    if (p / "lora_config.json").is_file() and (p / "lora_weights.safetensors").is_file():
        return "lora"
    if (p / "voxko_export.json").is_file():
        return "ft"
    if (p / "config.json").is_file():
        return "base"
    raise FileNotFoundError(f"VoxCPM2 모델 폴더가 아닙니다: {path}")


def ko_normalize(text: str) -> str:
    from ko_tts_norm import normalize
    return normalize(text, profile="qwen3tts")


class Engine:
    def __init__(self, model: str = DEFAULT_BASE, device: Optional[str] = None, base: Optional[str] = None,
                 emotion_map: Optional[str] = DEFAULT_MAP, warmup: bool = True):
        from voxcpm import VoxCPM
        self.path, self.kind = str(model), model_kind(model)
        kw = dict(load_denoiser=False, optimize=False, local_files_only=True, device=device)
        if self.kind == "lora":
            from voxcpm.model.voxcpm2 import LoRAConfig       # 🔴 V2
            cfg = json.loads(Path(model, "lora_config.json").read_text())
            self.base = base or cfg["base_model"]
            self.tts = VoxCPM.from_pretrained(self.base, lora_config=LoRAConfig(**cfg["lora_config"]), **kw)
            loaded, skipped = self.tts.tts_model.load_lora_weights(str(model))   # 다시 불러 개수 확인(같은 값 덮어씀)
            n_model = sum(1 for n, _ in self.tts.tts_model.named_parameters() if "lora_" in n)
            if skipped or len(loaded) != n_model:
                raise RuntimeError(f"LoRA 불러오기 불일치: 불러옴 {len(loaded)} · 건너뜀 {len(skipped)} · 모델 {n_model}")
            self.n_lora = len(loaded)
        else:
            self.base = None
            self.tts = VoxCPM.from_pretrained(self.path, **kw)
        self.sr = int(self.tts.tts_model.sample_rate)
        self.instruct = {}
        if emotion_map and os.path.isfile(emotion_map):
            import yaml
            self.instruct = (yaml.safe_load(open(emotion_map, encoding="utf-8")) or {}).get("instruct") or {}
        if warmup:
            self.synth("준비 합성입니다.", seed=0)

    def compose(self, text: str, emotion: Optional[str] = None, instruct: Optional[str] = None, normalize: bool = True) -> str:
        """정규화 + "(지시문)문장". emotion 은 표의 키, instruct 는 직접 쓴 지시문(둘 다 없으면 지시문 없음)."""
        t = ko_normalize(text) if normalize else text
        t = re.sub(r"\s+", " ", t).strip()
        ins = instruct
        if not ins and emotion:
            v = self.instruct[emotion]
            ins = v[0] if isinstance(v, list) else v           # 문구가 여러 개인 표(v2)는 1번 문구(= v1 과 같은 문구)
        return f"({ins}){t}" if ins else t

    def synth(self, text: str, ref: Optional[str] = None, seed: int = 1986, cfg_value: float = 2.0,
              inference_timesteps: int = 10, max_len: int = 4096, retry_badcase: bool = False) -> np.ndarray:
        """text 는 compose() 결과(이미 정규화 · 지시문 포함). ref = reference_wav_path(Controllable cloning)."""
        return self.tts.generate(text=text, reference_wav_path=ref, cfg_value=cfg_value,
                                 inference_timesteps=inference_timesteps, max_len=max_len, normalize=False,
                                 retry_badcase=retry_badcase, seed=seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_BASE, help="원본 · export_ft · export_lora 폴더")
    ap.add_argument("--base", default=None, help="LoRA 의 원본 경로(기본: lora_config.json 의 base_model)")
    ap.add_argument("--text", required=True)
    ap.add_argument("--emotion", default=None, help="지시문 표의 감정 키(예: Happy-B)")
    ap.add_argument("--instruct", default=None, help="지시문을 직접 줄 때")
    ap.add_argument("--emotion_map", default=DEFAULT_MAP)
    ap.add_argument("--ref", default=None, help="reference wav(목소리)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1986)
    ap.add_argument("--cfg_value", type=float, default=2.0)
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--no_normalize", action="store_true")
    a = ap.parse_args(argv)
    import soundfile as sf
    eng = Engine(a.model, base=a.base, emotion_map=a.emotion_map)
    text = eng.compose(a.text, a.emotion, a.instruct, normalize=not a.no_normalize)
    wav = eng.synth(text, ref=a.ref, seed=a.seed, cfg_value=a.cfg_value, inference_timesteps=a.timesteps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.out, wav, eng.sr)
    print(json.dumps({"out": a.out, "sec": round(len(wav) / eng.sr, 2), "kind": eng.kind, "model": eng.path,
                      "text": text, "ref": a.ref, "seed": a.seed}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
