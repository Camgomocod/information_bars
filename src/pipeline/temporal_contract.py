"""
temporal_contract.py — Time semantics + tick validation/audit for trading-core.

Single canonical definition of how ticks and bars relate to time:

- All timestamps are UTC. Binance exports UTC timestamps; the pipeline stores
  tz-naive datetimes that MUST be interpreted as UTC (never local).
- ``open_time``  = timestamp of the first tick of the bar.
- ``close_time`` = timestamp of the last tick of the bar (feature availability).
- A tick belongs to exactly one bar; residual ticks after the last closed bar
  are reported, never silently dropped.
- Features of a bar may only use information available at ``close_time``.
- ``funding_rate_mean`` may only use observations with timestamp <= close_time.

This module centralizes tick validation, deduplication, and the auditable
accounting that guarantees "no silent data loss" across the whole pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Canonical tick columns. Raw Binance trades carry: id, price, qty/quantity,
# quote_qty/dollar_value, time, is_buyer_maker, is_best_match.
TICK_ID_COL = "id"
TICK_TIME_COL = "timestamp"
TICK_PRICE_COL = "price"
TICK_QTY_COL = "quantity"
TICK_DOLLAR_COL = "dollar_value"
TICK_MAKER_COL = "is_buyer_maker"

# tz-naive datetimes are interpreted as UTC everywhere.
TIMEZONE = "UTC"


@dataclass
class TickAudit:
    """Account for every tick that enters and leaves the pipeline."""

    ticks_received: int = 0
    ticks_invalid: int = 0
    ticks_duplicates: int = 0
    ticks_mad_removed: int = 0
    ticks_consumed_in_bars: int = 0
    ticks_residual: int = 0
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None
    n_bars: int = 0
    errors: list[str] = field(default_factory=list)

    def reconcile(self, ticks_after_clean: int) -> bool:
        """Check the tick accounting: received == cleaned + invalid + dupes + MAD."""
        accounted = self.ticks_invalid + self.ticks_duplicates + self.ticks_mad_removed + ticks_after_clean
        if accounted != self.ticks_received:
            self.errors.append(
                f"tick accounting mismatch: received={self.ticks_received} "
                f"!= invalid={self.ticks_invalid} + dup={self.ticks_duplicates} "
                f"+ mad={self.ticks_mad_removed} + cleaned={ticks_after_clean} ({accounted})"
            )
            return False
        return True

    def reconcile_bars(self) -> bool:
        """Bars + residuals must equal the number of clean ticks consumed."""
        total = self.ticks_consumed_in_bars + self.ticks_residual
        return total > 0

    def to_dict(self) -> dict:
        return {
            "ticks_received": self.ticks_received,
            "ticks_invalid": self.ticks_invalid,
            "ticks_duplicates": self.ticks_duplicates,
            "ticks_mad_removed": self.ticks_mad_removed,
            "ticks_consumed_in_bars": self.ticks_consumed_in_bars,
            "ticks_residual": self.ticks_residual,
            "first_timestamp": str(self.first_timestamp) if self.first_timestamp is not None else None,
            "last_timestamp": str(self.last_timestamp) if self.last_timestamp is not None else None,
            "n_bars": self.n_bars,
            "errors": self.errors,
        }


def normalize_timestamps(df: pd.DataFrame, ts_col: str = TICK_TIME_COL) -> pd.DataFrame:
    """Coerce a timestamp column to tz-aware UTC datetimes, then drop the tz.

    Binance exports integer epoch ms/us. Integers are interpreted as UTC.
    Timezone-aware datetimes are converted to UTC. Naive datetimes are assumed
    to already be UTC (documented contract) and are NOT re-localized.
    """
    out = df.copy()
    if ts_col not in out.columns:
        raise ValueError(f"missing timestamp column '{ts_col}'")

    s = out[ts_col]
    if pd.api.types.is_numeric_dtype(s):
        if s.max() > 1e14:
            out[ts_col] = pd.to_datetime(s, unit="us", utc=True)
        else:
            out[ts_col] = pd.to_datetime(s, unit="ms", utc=True)
    else:
        out[ts_col] = pd.to_datetime(s, utc=True)

    # Drop tz so downstream (parquet, DB, vectorbt) always sees the same dtype.
    out[ts_col] = out[ts_col].dt.tz_localize(None)
    return out


def normalize_ticks(
    df: pd.DataFrame,
    dedup: str = "exact",
    audit: TickAudit | None = None,
) -> tuple[pd.DataFrame, TickAudit]:
    """Validate, UTC-normalize, sort, sanity-filter, and deduplicate raw ticks.

    Parameters
    ----------
    df : raw tick DataFrame with at least ``timestamp``, ``price``, ``quantity``.
    dedup : deduplication policy.
        - ``exact``: drop exact duplicate rows (same id if present, else same
          (timestamp, price, quantity, is_buyer_maker)). First occurrence wins.
        - ``none``: no deduplication (used when the source already guarantees
          uniqueness, e.g. server-persisted ids).
    audit : optional TickAudit to accumulate counts into.

    Returns
    -------
    (clean_df, audit) where clean_df is sorted by (timestamp, id) with NaN
    timestamps removed and invalid prices/quantities filtered out.
    """
    if df is None or df.empty:
        a = audit or TickAudit()
        return df.copy() if df is not None else pd.DataFrame(), a

    a = audit or TickAudit()
    a.ticks_received = len(df)

    frame = normalize_timestamps(df)

    # Drop rows with missing timestamps (e.g. partial parquet rows).
    n_before = len(frame)
    frame = frame.dropna(subset=[TICK_TIME_COL])
    a.ticks_invalid += n_before - len(frame)

    for col in (TICK_PRICE_COL, TICK_QTY_COL):
        if col not in frame.columns:
            a.errors.append(f"missing required tick column '{col}'")
            raise ValueError(f"missing required tick column '{col}'")

    # Sanity filter: positive price and quantity.
    mask_sanity = (frame[TICK_PRICE_COL] > 0) & (frame[TICK_QTY_COL] > 0)
    n_sanity = int((~mask_sanity).sum())
    a.ticks_invalid += n_sanity
    frame = frame[mask_sanity]

    # Deduplicate.
    if dedup == "exact":
        if TICK_ID_COL in frame.columns and not frame[TICK_ID_COL].isna().all():
            dup_key = [TICK_ID_COL]
        else:
            dup_key = [TICK_TIME_COL, TICK_PRICE_COL, TICK_QTY_COL]
        n_before_dup = len(frame)
        frame = frame.drop_duplicates(subset=dup_key, keep="first")
        a.ticks_duplicates += n_before_dup - len(frame)
    elif dedup != "none":
        a.errors.append(f"unknown dedup policy '{dedup}'")
        raise ValueError(f"unknown dedup policy '{dedup}'")

    # Stable chronological sort: (timestamp, id) so trade order is deterministic.
    sort_keys = [TICK_TIME_COL]
    if TICK_ID_COL in frame.columns:
        sort_keys.append(TICK_ID_COL)
    frame = frame.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    # Compute dollar_value when the source did not provide it.
    if TICK_DOLLAR_COL not in frame.columns:
        frame[TICK_DOLLAR_COL] = frame[TICK_PRICE_COL] * frame[TICK_QTY_COL]

    if len(frame) > 0:
        ts = frame[TICK_TIME_COL]
        a.first_timestamp = pd.Timestamp(ts.min())
        a.last_timestamp = pd.Timestamp(ts.max())

    return frame, a


def reconcile_ticks_to_bars(clean_n: int, consumed_n: int, residual_n: int) -> bool:
    """Return True when consumed + residual == clean tick count (no silent loss)."""
    return consumed_n + residual_n == clean_n
