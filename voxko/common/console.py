"""학습 CLI 출력 — colorama · logging · tqdm.

🔴 import 순서가 중요하다 (cosyvoice3_ko_ft trainer.py 에서 겪은 문제):
   colorama.init() 은 sys.stdout/sys.stderr 를 wrapper 로 **교체**한다. logging.basicConfig() 가 먼저 불리면
   그 StreamHandler 는 교체 전 stream 을 들고 있어, tqdm 의 logging_redirect_tqdm() 이 handler 를 찾지 못하고
   포맷 없는 handler 를 하나 더 붙인다 → 로그가 두 번 찍히고 진행바가 깨진다.
   → 이 모듈이 colorama.init → logging 설정 → tqdm import 를 **한 곳에서 이 순서로** 끝낸다.
   학습 진입점은 다른 무엇보다 먼저 `from voxko.common import console` 을 import 할 것.

색 규칙: 배너 CYAN · 시작/끝 GREEN · validation YELLOW · media BLUE · save MAGENTA · 경고 YELLOW · 오류 RED
colorama 가 없으면 색만 빠지고 동작은 같다. 환경변수 VOX_COLOR=1 이면 파이프·tee 출력에도 색을 남긴다.
"""
from __future__ import annotations

import logging
import sys
import textwrap
from typing import Iterable, Optional, Sequence, Tuple

try:
    import os as _os

    import colorama

    # VOX_COLOR=1 (scripts/run.sh 가 설정): 출력이 파이프(tee)여도 색을 유지 → 로그 파일을 less -R / tail -f 로 컬러로 본다
    _force = _os.environ.get("VOX_COLOR", "").lower() in ("1", "true", "yes", "always")
    # autoreset=False: c() 가 매번 RESET 을 붙이므로 충분. autoreset=True 면 tqdm·라이브러리의 **모든 write 뒤에**
    # RESET 코드가 붙어 로그 파일이 '\x1b[0m' 투성이가 된다(cv3_koft 실측).
    colorama.init(autoreset=False, strip=False if _force else None)
    Fore, Style, COLOR = colorama.Fore, colorama.Style, True
except Exception:  # noqa: BLE001 — colorama 없으면 색 없이
    COLOR = False

    class _NoColor:
        def __getattr__(self, _):
            return ""

    Fore = Style = _NoColor()  # type: ignore


class _ColorFormatter(logging.Formatter):
    _LEVEL = {logging.WARNING: "YELLOW", logging.ERROR: "RED", logging.CRITICAL: "RED"}

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        color = self._LEVEL.get(record.levelno)
        return f"{getattr(Fore, color)}{msg}{Style.RESET_ALL}" if (COLOR and color) else msg


class _DropNoise(logging.Filter):
    """무해하지만 반복되는 라이브러리 메시지를 숨긴다.

    · fp32 master weight + bf16 autocast 에서 flash-attn 이 입력을 bf16 으로 되돌린다는 안내(정상 동작)
    """
    _PATTERNS = ("Flash Attention 2 only supports", "Casting fp32 inputs back to torch.bfloat16",
                 "attempting to use Flash Attention 2 without specifying a torch dtype",
                 "Setting `pad_token_id` to `eos_token_id`")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._PATTERNS)


_handler = logging.StreamHandler(sys.stderr)          # colorama 교체 **후** 의 stderr
_handler.setFormatter(_ColorFormatter("%(asctime)s %(levelname).1s %(message)s", datefmt="%H:%M:%S"))
_handler.addFilter(_DropNoise())
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger("voxko")
# 모델 로드 때 transformers 의 INFO 가 많다 → WARNING 이상만
for _name in ("transformers", "datasets"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from tqdm.auto import tqdm  # noqa: E402,F401  (반드시 colorama·logging 설정 뒤)
from tqdm.contrib.logging import logging_redirect_tqdm  # noqa: E402,F401


def c(text, *styles: str) -> str:
    """c('loss', 'CYAN', 'BRIGHT') — Fore/Style 이름으로 색을 입힌다."""
    if not COLOR or not styles:
        return str(text)
    pre = "".join(getattr(Fore, s, "") or getattr(Style, s, "") for s in styles)
    return f"{pre}{text}{Style.RESET_ALL}"


def is_main_env() -> bool:
    """accelerate 를 만들기 전에도 쓸 수 있는 rank0 판정(torchrun/accelerate 가 RANK 를 넣는다)."""
    import os
    return os.environ.get("RANK", "0") == "0"


def say(msg: str, *styles: str) -> None:
    """rank0 에서만 print (accelerate 생성 전 정책 메시지용)."""
    if is_main_env():
        print(c(msg, *styles), flush=True)


def set_main_process(is_main: bool) -> None:
    """rank0 만 INFO, 나머지 rank 는 WARNING 이상만 출력(로그 중복 방지).

    transformers 는 자기 logger 에 별도 handler 를 달아 root 필터를 우회한다 → 끄고 root 로 전파시켜
    같은 형식·같은 필터(_DropNoise)를 거치게 한다. 비 rank0 의 transformers 경고도 ERROR 이상만.
    """
    logging.getLogger().setLevel(logging.INFO if is_main else logging.WARNING)
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.disable_default_handler()
        hf_logging.enable_propagation()
        hf_logging.set_verbosity_warning() if is_main else hf_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001
        pass


def banner(title: str, sections: Sequence[Tuple[str, Iterable[Tuple[str, object]]]], width: int = 0) -> None:
    """섹션별 key : value 표를 박스로 출력한다.

    banner("VoxCPM2 FT", [("Run", [("name", "S00"), ...]), ("Optim", [...])])
    """
    import shutil
    width = width or min(max(shutil.get_terminal_size((100, 20)).columns, 80), 120)
    print()
    print(c("╔" + "═" * (width - 2) + "╗", "CYAN"))
    print(c("║ " + f"🎧  {title}".ljust(width - 4) + " ║", "CYAN", "BRIGHT"))
    for name, rows in sections:
        print(c("╟" + "─" * (width - 2) + "╢", "CYAN"))
        print(c("║ ", "CYAN") + c(f"[{name}]", "CYAN", "BRIGHT"))
        for k, v in rows:
            # 긴 값은 공백에서만 다음 줄로(경로는 자르지 않는다)
            chunks = textwrap.wrap(str(v), width - 24, break_long_words=False, break_on_hyphens=False) or [""]
            print(c("║ ", "CYAN") + c(f"  {k:<14}", "BRIGHT") + f" : {chunks[0]}")
            for more in chunks[1:]:
                print(c("║ ", "CYAN") + " " * 19 + more)
    print(c("╚" + "═" * (width - 2) + "╝", "CYAN"))


def rule(text: str = "", color: str = "GREEN", width: int = 84) -> None:
    """━━━ text ━━━ 구분선."""
    t = f" {text} " if text else ""
    side = max((width - len(t)) // 2, 3)
    print(c("━" * side + t + "━" * side, color, "BRIGHT"))


def fmt_metrics(d: dict, keys: Optional[Sequence[str]] = None, digits: int = 4) -> str:
    """{'loss': 1.234567, 'lr': 1e-5} → 'loss 1.2346 | lr 1.0e-05'."""
    out = []
    for k in keys or d.keys():
        if k not in d or d[k] is None:
            continue
        v = d[k]
        if isinstance(v, float):
            v = f"{v:.1e}" if (abs(v) < 1e-3 and v != 0) else f"{v:.{digits}f}"
        out.append(f"{k} {v}")
    return " | ".join(out)


def fmt_seconds(s: float) -> str:
    s = int(max(s, 0))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
