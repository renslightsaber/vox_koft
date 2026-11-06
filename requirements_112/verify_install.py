"""vox_koft env "voxcpm" 검증 — import · GPU · VoxCPM2 로드 · 한국어 합성 1개(서비스 ref)
    CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/voxcpm/bin/python requirements_112/verify_install.py
결과 wav: ~/workspaces/vox_koft/_verify/verify_ko.wav"""
import os, time, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
print(f"python {sys.version.split()[0]} · torch {torch.__version__} · CUDA {torch.version.cuda} · GPU {torch.cuda.is_available()}"
      + (f" · {torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}" if torch.cuda.is_available() else ""))
import transformers, torchaudio, voxcpm, ko_tts_norm
print(f"transformers {transformers.__version__} · torchaudio {torchaudio.__version__} · voxcpm {getattr(voxcpm, '__version__', '?')} · ko_tts_norm OK")
from voxcpm import VoxCPM
CK = Path.home() / "workspaces/ourckpts/voxcpm/VoxCPM2"
REF = Path.home() / "workspaces/reference_audios/qwen3tts/set2_trim03_rms20/female_ment_natural_24k_trim03_rms20.wav"
REF_TEXT = "모르는 문제가 있으면 그냥 넘어가지 말고 꼭 질문해줘. 틀린 문제를 다시 푸는 게 실력을 키우는 가장 좋은 방법이야."
t0 = time.time()
m = VoxCPM.from_pretrained(str(CK), load_denoiser=False, optimize=False, local_files_only=True)
print(f"모델 로드 {time.time() - t0:.1f}s · 출력 SR {m.tts_model.sample_rate}")
text = ko_tts_norm.normalize("밀크T로 공부하면서 더 궁금한 것이 있나요? 편하게 질문해 주세요.", profile="qwen3tts")
t0 = time.time()
w = m.generate(text=text, prompt_wav_path=str(REF), prompt_text=REF_TEXT, reference_wav_path=str(REF), seed=1986)
dt = time.time() - t0; sr = m.tts_model.sample_rate; d = len(w) / sr
out = Path.home() / "workspaces/vox_koft/_verify"; out.mkdir(exist_ok=True); sf.write(out / "verify_ko.wav", w, sr)
print(f"입력: {text}\n합성 {d:.2f}s · {dt:.1f}s 걸림(RTF {dt / d:.2f}) · 최대 VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f}GB · 저장 {out / 'verify_ko.wav'}")
assert 1.0 < d < 20 and np.isfinite(w).all(), "합성 길이 · 값 이상"
print("✅ verify OK")
