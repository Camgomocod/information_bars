"""
BarBuilder — Downloads tick data for a month and converts to Dollar Run Bars (DRBs)

Pipeline per month:
  1. For each day in YYYY-MM: download ticks via DownloadData.download_day()
  2. Clean ticks with DataNormalizer.clean_raw_ticks()
  3. Concatenate all days → full month tick DataFrame
  4. Convert to DRBs using OptimizedInfoRunBars.get_drbs() with provided hyperparams
  5. Return bars DataFrame (OHLCV + metadata)
"""

import sys
from pathlib import Path
from datetime import date, timedelta
from calendar import monthrange
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Ensure project root is on path even when called as script
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.connectors.download_data import DownloadData
from src.bars.info_bars import OptimizedInfoRunBars
from src.normalizers.data_normalizer import DataNormalizer
from src.pipeline.temporal_contract import TickAudit


def build_monthly_daily_bars(
    symbol: str,
    year: int,
    month: int,
    params: Dict,
    min_days: int = 15,
    min_bars: int = 40,
    verbose: bool = True,
    source: str = "binance",
    data_dir: str = "data_raw",
) -> Optional[pd.DataFrame]:
    """Build DRBs independently per day while retaining day provenance.

    The optimizer evaluates one day at a time. The yearly builder must use the
    same unit; generating one DRB stream from a whole month makes it impossible
    to know which base bars belong to failed days and causes recovery bars to
    be misclassified or duplicated.
    """
    downloader = DownloadData(symbol=symbol, source=source, data_dir=data_dir)
    normalizer = DataNormalizer()
    n_days = monthrange(year, month)[1]
    start = date(year, month, 1)
    day_frames: list[pd.DataFrame] = []
    day_audits: list[dict] = []
    loaded_days = 0

    # Local monthly parquets are already partitioned by download_date. Read
    # them once instead of issuing one predicate read per calendar day.
    local_days: dict[str, pd.DataFrame] = {}
    monthly_path = Path(data_dir) / str(year) / f"{symbol}_{year}_{month:02d}.parquet"
    if source in ("local", "auto") and monthly_path.exists():
        try:
            monthly_raw = pd.read_parquet(monthly_path)
            if "download_date" in monthly_raw.columns:
                for day, group in monthly_raw.groupby(monthly_raw["download_date"].astype(str)):
                    local_days[str(day)[:10]] = group
        except Exception as exc:
            if verbose:
                print(f"   ⚠️  Could not cache {monthly_path.name}: {exc}")

    for offset in range(n_days):
        date_str = (start + timedelta(days=offset)).strftime("%Y-%m-%d")
        raw = local_days.get(date_str)
        if raw is None:
            raw = downloader.download_day(date_str)
        if raw is None or raw.empty:
            continue
        loaded_days += 1
        audit = TickAudit(ticks_received=len(raw))
        invalid_mask = (raw["price"] <= 0) | (raw["quantity"] <= 0)
        audit.ticks_invalid = int(invalid_mask.sum())
        needed = [c for c in ("timestamp", "price", "quantity", "dollar_value") if c in raw.columns]
        clean = normalizer.clean_raw_ticks(raw[needed].copy(), mad_window=100, k=10.0)
        audit.ticks_mad_removed = max(
            0, len(raw) - audit.ticks_invalid - len(clean)
        ) if clean is not None else 0
        if clean is None or clean.empty:
            day_audits.append({
                "date": date_str, "source": "base", "ticks_received": len(raw),
                "ticks_invalid": audit.ticks_invalid, "ticks_mad_removed": audit.ticks_mad_removed,
                "ticks_clean": 0, "ticks_consumed": 0, "ticks_residual": 0,
                "bars": 0, "errors": "empty_after_clean",
            })
            continue
        if "dollar_value" not in clean.columns:
            clean["dollar_value"] = clean["price"] * clean["quantity"]

        bars, bar_audit = OptimizedInfoRunBars(save_path=str(project_root / "data_optimized" / "tmp")).get_drbs_audited(
            df=clean,
            exp_lambda=params["exp_lambda"],
            init_exp_T=params["init_exp_T"],
        )
        day_audits.append({
            "date": date_str, "source": "base", "ticks_received": len(raw),
            "ticks_invalid": audit.ticks_invalid, "ticks_mad_removed": audit.ticks_mad_removed,
            "ticks_clean": len(clean), "ticks_consumed": bar_audit.consumed_ticks,
            "ticks_residual": bar_audit.residual_ticks, "bars": len(bars),
            "errors": "; ".join(bar_audit.violations),
        })
        if bars is None or bars.empty:
            continue

        bars = bars.copy()
        bars["_bar_date"] = date_str
        bars["_base_failed_day"] = np.int8(len(bars) < min_bars)
        bars["_recovered_bar"] = False
        day_frames.append(bars)

    if loaded_days < min_days or not day_frames:
        if verbose:
            print(
                f"   ⚠️  [{symbol}] {year}-{month:02d}: "
                f"{loaded_days} loaded days, minimum {min_days}"
            )
        return None

    result = pd.concat(day_frames, ignore_index=True)
    result = result.sort_values("open_time").reset_index(drop=True)
    result.attrs["day_audits"] = day_audits
    return result


def build_monthly_bars(
    symbol: str,
    year: int,
    month: int,
    params: Dict,
    min_days: int = 15,
    verbose: bool = True,
    source: str = "binance",
    data_dir: str = "data_raw",
) -> Optional[pd.DataFrame]:
    """
    Downloads all available daily tick data for YYYY-MM and converts to DRBs.

    Parameters
    ----------
    symbol    : e.g. "BTCUSDT"
    year      : e.g. 2023
    month     : e.g. 1
    params    : {"exp_lambda": float, "init_exp_T": int}
    min_days  : minimum successful days required (skip month if below threshold)
    verbose   : print progress
    source    : "binance" (download) or "local" (parquet files)
    data_dir  : directory where raw tick parquet files are stored

    Returns
    -------
    pd.DataFrame with bar columns, or None if not enough data.
    """
    downloader = DownloadData(symbol=symbol, source=source, data_dir=data_dir)
    normalizer = DataNormalizer()

    # Fast skip: when not explicitly downloading from Binance, a month without
    # its monthly raw parquet has NO local data. Skip BEFORE the day loop so we
    # never hit the per-day Binance retry hang on nonexistent days (e.g. future
    # months of a partial year).
    if source in ("local", "auto"):
        monthly_parquet = Path(data_dir) / str(year) / f"{symbol}_{year}_{month:02d}.parquet"
        if not monthly_parquet.exists():
            if verbose:
                print(f"   ⏭️  [{symbol}] {year}-{month:02d}: no raw data "
                      f"({monthly_parquet.name} missing) → skipping month")
            return None
        # Partial months (e.g. current month only has a few days published):
        # counting present days avoids a per-day Binance fallback loop for a
        # month that can never reach min_days.
        try:
            present = len(
                pd.read_parquet(monthly_parquet, columns=["download_date"])["download_date"].drop_duplicates()
            )
        except Exception:
            present = None
        if present is not None and present < min_days:
            if verbose:
                print(f"   ⏭️  [{symbol}] {year}-{month:02d}: only {present} day(s) in raw parquet "
                      f"(min={min_days}) → skipping month")
            return None

    # Temporary save path for bars (not actually used for save, just required by constructor)
    bars_gen = OptimizedInfoRunBars(save_path=str(project_root / "data_optimized" / "tmp"))

    n_days = monthrange(year, month)[1]
    start = date(year, month, 1)

    tick_chunks: list[pd.DataFrame] = []
    successful_days = 0
    failed_days = 0

    if verbose:
        print(f"\n📅 [{symbol}] Building bars for {year}-{month:02d} "
              f"(λ={params['exp_lambda']:.4f}, T={params['init_exp_T']})...")

    for day_offset in range(n_days):
        current_date = start + timedelta(days=day_offset)
        date_str = current_date.strftime("%Y-%m-%d")

        df_raw = downloader.download_day(date_str)

        if df_raw is None or df_raw.empty:
            failed_days += 1
            continue

        # Keep only necessary columns (delay cleaning to month-level for efficiency)
        needed = ["timestamp", "price", "quantity"]
        if "dollar_value" in df_raw.columns:
            needed.append("dollar_value")

        df_raw = df_raw[needed].copy()
        tick_chunks.append(df_raw)
        successful_days += 1

    if successful_days < min_days:
        if verbose:
            print(f"   ⚠️  [{symbol}] {year}-{month:02d}: Only {successful_days} days loaded "
                  f"(min={min_days}) → skipping month")
        return None

    if verbose:
        print(f"   📦 [{symbol}] {year}-{month:02d}: {successful_days}/{n_days} days loaded, "
              f"concatenating ticks...")

    # Concatenate and sort by timestamp
    all_ticks = pd.concat(tick_chunks, ignore_index=True)
    all_ticks = all_ticks.sort_values("timestamp").reset_index(drop=True)

    # Apply MAD outlier filter once on the full month (much faster than day-by-day)
    all_ticks = normalizer.clean_raw_ticks(all_ticks, mad_window=100, k=10.0)

    if all_ticks is None or all_ticks.empty:
        if verbose:
            print(f"   ❌ [{symbol}] {year}-{month:02d}: All ticks removed by outlier filter")
        return None

    # Ensure dollar_value column exists for the bar generator
    if "dollar_value" not in all_ticks.columns:
        all_ticks["dollar_value"] = all_ticks["price"] * all_ticks["quantity"]

    total_ticks = len(all_ticks)
    if verbose:
        print(f"   🔢 [{symbol}] {year}-{month:02d}: {total_ticks:,} total ticks → generating DRBs...")

    # Generate Dollar Run Bars
    bars_df = bars_gen.get_drbs(
        df=all_ticks,
        exp_lambda=params["exp_lambda"],
        init_exp_T=params["init_exp_T"],
    )

    if bars_df is None or bars_df.empty:
        if verbose:
            print(f"   ❌ [{symbol}] {year}-{month:02d}: No bars generated")
        return None

    # Attach hyperparams as metadata columns (useful for training traceability)
    bars_df["exp_lambda"] = np.float32(params["exp_lambda"])
    bars_df["init_exp_T"] = np.int32(params["init_exp_T"])

    # Convert timestamps to datetime64 if they are numpy ints (from numba path)
    for col in ["open_time", "close_time"]:
        if col in bars_df.columns and bars_df[col].dtype != "datetime64[ns]":
            bars_df[col] = pd.to_datetime(bars_df[col])

    # Downcast numeric columns to float32 to save memory
    float_cols = ["open", "high", "low", "close", "volume", "dollar_value"]
    for col in float_cols:
        if col in bars_df.columns:
            bars_df[col] = bars_df[col].astype(np.float32)

    if "n_ticks" in bars_df.columns:
        bars_df["n_ticks"] = bars_df["n_ticks"].astype(np.int32)

    if verbose:
        print(f"   ✅ [{symbol}] {year}-{month:02d}: {len(bars_df):,} DRBs generated")

    return bars_df
