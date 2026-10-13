"""
Strategy parameter loading for the options auto-trader.

Every entry/exit threshold used by the scanners and agents is read from a TOML
file at import time rather than hardcoded in the sources. This keeps tradeable
parameters out of version control: `config/strategy.toml` is gitignored, and
only `config/strategy.example.toml` (neutral placeholders) is committed.

Resolution order:
  1. $AUTOTRADER_STRATEGY_CONFIG, if set
  2. config/strategy.toml
  3. config/strategy.example.toml  -- placeholders only, warns on use

Usage:
    from strategy_params import PARAMS
    if vol_ratio >= PARAMS["breakout"]["min_vol_ratio"]:
        ...
"""

from __future__ import annotations

import os
import tomllib
import warnings
from datetime import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_CONFIG_DIR = _REPO_ROOT / "config"

REAL_CONFIG = _CONFIG_DIR / "strategy.toml"
EXAMPLE_CONFIG = _CONFIG_DIR / "strategy.example.toml"


class StrategyConfigError(RuntimeError):
    """Raised when no strategy configuration can be loaded."""


def _resolve_path() -> tuple[Path, bool]:
    """Return (path, is_placeholder)."""
    override = os.environ.get("AUTOTRADER_STRATEGY_CONFIG")
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise StrategyConfigError(
                f"AUTOTRADER_STRATEGY_CONFIG points at {p}, which does not exist."
            )
        return p, False

    if REAL_CONFIG.is_file():
        return REAL_CONFIG, False

    if EXAMPLE_CONFIG.is_file():
        return EXAMPLE_CONFIG, True

    raise StrategyConfigError(
        f"No strategy configuration found. Copy {EXAMPLE_CONFIG.name} to "
        f"{REAL_CONFIG.name} in {_CONFIG_DIR} and fill in your parameters."
    )


def load_params(path: Path | None = None) -> dict:
    if path is None:
        path, is_placeholder = _resolve_path()
    else:
        is_placeholder = False

    with path.open("rb") as fh:
        params = tomllib.load(fh)

    if is_placeholder:
        warnings.warn(
            "Loaded PLACEHOLDER strategy parameters from "
            f"{path.name}. These are documentation values, not a tradeable "
            "configuration -- signals produced with them are meaningless. "
            "Copy it to config/strategy.toml and set real values.",
            RuntimeWarning,
            stacklevel=2,
        )

    params["_source"] = str(path)
    params["_is_placeholder"] = is_placeholder
    return params


def parse_hhmm(value: str) -> time:
    """Parse a "HH:MM" config string into a datetime.time."""
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


PARAMS = load_params()
