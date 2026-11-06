"""저장소 경로 해석.

⚠️ 이 모듈은 stdlib 만 import 하고 logging·stdout 을 건드리지 않는다.
   (console.py 의 colorama → logging 초기화 순서를 깨지 않기 위해)
"""
from __future__ import annotations

import os
from pathlib import Path

_SENTINELS = ("VoxCPM", "configs", "voxko")


def find_repo_root() -> Path:
    """환경변수 VOX_ROOT 가 있으면 그 값, 없으면 sentinel 세 개가 모두 있는 상위 디렉터리."""
    env = os.environ.get("VOX_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if all((p / s).exists() for s in _SENTINELS):
            return p
    raise RuntimeError(f"저장소 루트를 찾지 못했습니다({_SENTINELS}). VOX_ROOT 환경변수로 지정하세요.")


REPO_ROOT = find_repo_root()
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG = CONFIGS_DIR / "config.yaml"


def resolve(path: str | os.PathLike, base: str | os.PathLike | None = None) -> Path:
    """~ 확장 후, 상대경로면 base(기본: 저장소 루트) 기준으로 푼다."""
    p = Path(os.path.expanduser(str(path)))
    if not p.is_absolute():
        p = Path(base or REPO_ROOT) / p
    return p
