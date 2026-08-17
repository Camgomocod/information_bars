"""
bar_audit.py — Invariant checks and tick-accounting for generated bars.

The DRB builder must satisfy these invariants:

- Bars are strictly ordered by open_time.
- open_time <= close_time for every bar.
- OHLC is valid: high >= max(open, close), low <= min(open, close), high >= low.
- Bar boundaries are contiguous and non-overlapping: every clean tick belongs
  to at most one bar, and no tick is silently dropped between bars.
- Consumed ticks + residual ticks == clean ticks fed to the builder.

``check_bar_invariants`` returns a list of violations (empty = valid).
``BarAudit`` combines tick accounting with bar invariants into a single report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BAR_TIME_COLS = ("open_time", "close_time")


@dataclass
class BarAudit:
    """Tick/bar accounting for a single DRB generation call."""

    clean_ticks: int = 0
    consumed_ticks: int = 0
    residual_ticks: int = 0
    n_bars: int = 0
    violations: list[str] = field(default_factory=list)

    @property
    def no_silent_loss(self) -> bool:
        return self.consumed_ticks + self.residual_ticks == self.clean_ticks

    def to_dict(self) -> dict:
        return {
            "clean_ticks": self.clean_ticks,
            "consumed_ticks": self.consumed_ticks,
            "residual_ticks": self.residual_ticks,
            "n_bars": self.n_bars,
            "no_silent_loss": self.no_silent_loss,
            "violations": self.violations,
        }


def check_bar_invariants(bars: pd.DataFrame) -> list[str]:
    """Return a list of invariant violations (empty list == valid bars)."""
    violations: list[str] = []
    if bars is None or bars.empty:
        return ["empty bars"]

    for col in BAR_TIME_COLS:
        if col not in bars.columns:
            violations.append(f"missing column '{col}'")
            return violations

    open_t = pd.to_datetime(bars["open_time"])
    close_t = pd.to_datetime(bars["close_time"])

    # 1. Ordered bars.
    if not open_t.is_monotonic_increasing:
        violations.append("bars not ordered by open_time")

    # 2. open <= close per bar.
    bad_duration = (close_t < open_t).sum()
    if bad_duration > 0:
        violations.append(f"{bad_duration} bars with close_time < open_time")

    # 3. No overlap between consecutive bars.
    if len(bars) > 1:
        nxt_open = open_t.iloc[1:].reset_index(drop=True)
        prev_close = close_t.iloc[:-1].reset_index(drop=True)
        n_overlap = int((nxt_open < prev_close).sum())
        if n_overlap > 0:
            violations.append(f"{n_overlap} overlapping consecutive bars")

    # 4. OHLC validity.
    if {"open", "high", "low", "close"}.issubset(bars.columns):
        o = bars["open"].to_numpy()
        h = bars["high"].to_numpy()
        lo = bars["low"].to_numpy()
        c = bars["close"].to_numpy()
        if (h < lo).any():
            violations.append("high < low in some bars")
        if (h < np.maximum(o, c)).any():
            violations.append("high < max(open, close) in some bars")
        if (lo > np.minimum(o, c)).any():
            violations.append("low > min(open, close) in some bars")

    # 5. Non-negative activity and positive tick counts.
    for col in ("n_ticks", "volume", "dollar_value"):
        if col in bars.columns:
            bad = (bars[col] < 0).sum() if col == "n_ticks" else (bars[col].isna() | (bars[col] < 0)).sum()
            if bad > 0:
                violations.append(f"{bad} bars with invalid {col} (< 0 or NaN)")
    if "n_ticks" in bars.columns and (bars["n_ticks"] <= 0).any():
        violations.append("bars with n_ticks <= 0")

    return violations
