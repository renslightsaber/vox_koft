"""오디오 유틸: mono 로드(+ resample) · log-mel · TensorBoard 용 mel figure.

VoxCPM2: 학습 입력(AudioVAE encoder) = 16kHz · 출력(decoder) = 48kHz.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

SR_IN = 16000      # 학습 입력 · VAE encoder
SR_OUT = 48000     # 합성 출력 · VAE decoder


def load_audio(path: str, sr: int = SR_IN) -> np.ndarray:
    """wav 를 float32 mono, sr 로 읽는다(필요하면 librosa 로 resample)."""
    wav, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def mel(wav: np.ndarray | torch.Tensor, sr: int) -> np.ndarray:
    """(T,) 파형 → [80, frames] log-mel (그림용, 표시 목적이라 sr 별로 hop 을 맞춤)."""
    import librosa
    y = np.asarray(wav, dtype=np.float32).reshape(-1)
    hop = sr // 100                                   # 10ms
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=4 * hop * 2, hop_length=hop, n_mels=80, fmax=min(sr // 2, 12000))
    return np.log(np.clip(m, 1e-5, None))


def mel_to_figure(m, title: str | None = None):
    """[n_mels, T] mel → matplotlib Figure (TB add_figure(close=True) 용)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = np.asarray(m)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(m, origin="lower", aspect="auto", cmap="magma")
    if title:
        ax.set_title(title)
    ax.set_xlabel("frame (10ms)")
    ax.set_ylabel("mel")
    fig.tight_layout()
    return fig
