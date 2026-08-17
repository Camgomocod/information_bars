#!/usr/bin/env python3
"""
build_yearly_training_data.py — Genera parquet de training por año y símbolo

Este script integra todo el pipeline de generación de datos de training:
1. Procesa meses con hyperparams optimizados (walk-forward)
2. Detecta días que fallan (< 50 bars)
3. Si no existen params para días fallidos, corre optimización inline automáticamente
4. Recupera días fallidos con params especializados
5. Genera parquet final con columnas failed_day y sample_weight

Usage:
    micromamba run -n trading-core python scripts/build_yearly_training_data.py \
        --year 2023

    micromamba run -n trading-core python scripts/build_yearly_training_data.py \
        --start-year 2023 --end-year 2025

    micromamba run -n trading-core python scripts/build_yearly_training_data.py \
        --year 2025 --symbols BTCUSDT ETHUSDT --dry-run

    # Con trials personalizados para optimización de días fallidos
    micromamba run -n trading-core python scripts/build_yearly_training_data.py \
        --year 2023 --trials 50 --source local --data-dir data_raw
"""

import sys
import argparse
from pathlib import Path
from datetime import date, timedelta
from calendar import monthrange
import warnings
import gc
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline.hyperparam_loader import HyperparamLoader
from src.pipeline.bar_builder import build_monthly_daily_bars
from src.features.base_features import compute_all_features
from src.features.positioning_calculator import PositioningCalculator
from src.connectors.download_data import DownloadData
from src.bars.info_bars import OptimizedInfoRunBars
from src.bars.bars_statistics import BarsStatistics
from src.normalizers.data_normalizer import DataNormalizer
from src.pipeline.data_validation import write_validated_parquet
from src.pipeline.quality_report import (
    daily_audit_frame,
    monthly_quality_report,
    validate_study_metadata,
    validate_daily_audit,
    validate_monthly_quality,
    is_partial_month,
)


SAMPLE_WEIGHT_NORMAL = 1.0
SAMPLE_WEIGHT_FAILED = 0.5
MIN_BARS_THRESHOLD = 40
FAILED_DAYS_MIN_TO_OPTIMIZE = 3
FAILED_DAYS_OPTIMIZE_TRIALS = 20


def get_failed_days_optimized_params(experiments_dir: Path, symbol: str, year: int, month: int) -> dict | None:
    """Load optimized params for failed days if they exist."""
    failed_dir = experiments_dir / f"failed_days_{year}_{month:02d}" / symbol
    params_file = failed_dir / f"{symbol}_failed_best_params.yaml"

    if params_file.exists():
        with open(params_file) as f:
            params = yaml.safe_load(f)
            return {
                "exp_lambda": params["exp_lambda"],
                "init_exp_T": params["init_exp_T"],
            }
    return None


def optimize_failed_days_inline(
    symbol: str,
    year: int,
    month: int,
    failed_dates: list,
    experiments_dir: Path,
    n_trials: int = 30,
    verbose: bool = True,
    source: str = "binance",
    data_dir: str = "data_raw",
) -> dict | None:
    """
    Download ticks for failed days and run inline optimization.
    Saves params to experiments/failed_days_YYYY_MM/symbol/symbol_failed_best_params.yaml
    """
    if verbose:
        print(f"\n  🔬 Inline optimization for {len(failed_dates)} failed days: {symbol} {year}-{month:02d}")

    if len(failed_dates) == 0:
        return None

    downloader = DownloadData(symbol=symbol, source=source, data_dir=data_dir)
    normalizer = DataNormalizer()
    bars_gen = OptimizedInfoRunBars(save_path=str(project_root / "data_optimized" / "tmp"))

    ticks_by_date = {}
    for date_str in failed_dates:
        try:
            df_raw = downloader.download_day(date_str)
            if df_raw is None or df_raw.empty:
                continue

            df_clean = normalizer.clean_raw_ticks(df_raw, mad_window=100, k=10.0)
            if df_clean is None or df_clean.empty:
                continue

            ticks_by_date[date_str] = df_clean
            if verbose:
                print(f"     ✅ {date_str}: {len(df_clean):,} ticks")
        except Exception as e:
            if verbose:
                print(f"     ❌ {date_str}: Error - {e}")

    if len(ticks_by_date) == 0:
        if verbose:
            print(f"     ⚠️  No ticks downloaded")
        return None

    loader = HyperparamLoader(experiments_dir)

    if month == 1:
        ref_month = 12
        ref_year = year - 1
    else:
        ref_month = month - 1
        ref_year = year

    hypermeta = loader.get_params_with_meta(symbol, ref_year, ref_month, verbose=False)

    base_lambda = hypermeta.exp_lambda
    base_T = hypermeta.init_exp_T

    T_min = max(100, int(base_T * 0.5))
    T_max = max(T_min + 100, int(base_T * 1.5))

    if verbose:
        print(f"     📊 Previous params: λ={base_lambda:.4f}, T={base_T}")
        print(f"     📊 Search bounds: T=[{T_min}, {T_max}], λ=[0.990, 0.999]")
        print(f"     🔄 Running {n_trials} trials...")

    study = optuna.create_study(
        study_name=f"{symbol}_failed_{year}_{month:02d}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial: optuna.Trial) -> float:
        T = trial.suggest_int("init_exp_T", T_min, T_max)
        exp_lambda = trial.suggest_float("exp_lambda", 0.990, 0.999)

        successful = 0
        total = len(ticks_by_date)
        quality_scores = []

        for date_str, ticks in ticks_by_date.items():
            try:
                bars = bars_gen.get_drbs(ticks, exp_lambda=exp_lambda, init_exp_T=T)
                if bars is not None and len(bars) >= MIN_BARS_THRESHOLD:
                    successful += 1
                    try:
                        quality_scores.append(BarsStatistics().compute_quality_score_fast(bars))
                    except Exception:
                        quality_scores.append(0.0)
            except Exception:
                continue

        if successful == 0:
            return 0.0

        success_rate = successful / total
        if success_rate < 0.4:
            return 0.0

        mean_quality = float(np.mean(quality_scores)) if quality_scores else 0.0
        quality_stability = (
            max(0.0, 100.0 - float(np.std(quality_scores)))
            if quality_scores else 0.0
        )
        # Recovery must preserve useful sampling characteristics, not merely
        # cross the minimum-bar threshold on a subset of days.
        objective_score = 0.50 * success_rate * 100 + 0.35 * mean_quality + 0.15 * quality_stability
        trial.set_user_attr("recovery_success_rate", success_rate)
        trial.set_user_attr("mean_quality_score", mean_quality)
        return objective_score

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if verbose:
        print(f"     📊 Best value: {study.best_value:.2f}% ({len(study.trials)} trials)")

    if study.best_trial:
        best_params = study.best_trial.params

        exp_dir = experiments_dir / f"failed_days_{year}_{month:02d}" / symbol
        exp_dir.mkdir(parents=True, exist_ok=True)

        params_file = exp_dir / f"{symbol}_failed_best_params.yaml"
        with open(params_file, 'w') as f:
            yaml.dump({
                "symbol": symbol,
                "year": year,
                "month": month,
                "exp_lambda": best_params['exp_lambda'],
                "init_exp_T": best_params['init_exp_T'],
                "objective_score": float(study.best_value),
                "success_rate": float(study.best_trial.user_attrs.get("recovery_success_rate", 0.0)),
                "mean_quality_score": float(study.best_trial.user_attrs.get("mean_quality_score", 0.0)),
                "n_failed_days": len(ticks_by_date),
                "n_trials": n_trials,
            }, f)

        if verbose:
            print(f"     💾 Saved to: {params_file}")
            print(f"     ✅ Best: λ={best_params['exp_lambda']:.4f}, T={best_params['init_exp_T']}")

        return {
            "exp_lambda": best_params['exp_lambda'],
            "init_exp_T": best_params['init_exp_T'],
        }

    return None


def reprocess_failed_day(symbol: str, date_str: str, params: dict, source: str = "binance", data_dir: str = "data_raw") -> pd.DataFrame | None:
    """Download ticks and generate DRBs for a single failed day with custom params."""
    try:
        downloader = DownloadData(symbol=symbol, source=source, data_dir=data_dir)
        normalizer = DataNormalizer()
        bars_gen = OptimizedInfoRunBars(save_path=str(project_root / "data_optimized" / "tmp"))

        df_raw = downloader.download_day(date_str)
        if df_raw is None or df_raw.empty:
            return None

        df_clean = normalizer.clean_raw_ticks(df_raw, mad_window=100, k=10.0)
        if df_clean is None or df_clean.empty:
            return None

        bars = bars_gen.get_drbs(
            df_clean,
            exp_lambda=params["exp_lambda"],
            init_exp_T=params["init_exp_T"],
        )

        if bars is not None and len(bars) >= MIN_BARS_THRESHOLD:
            bars = bars.copy()
            bars["_bar_date"] = date_str
            return bars

        return None
    except Exception:
        return None


def reprocess_failed_days(
    symbol: str,
    failed_dates: list,
    year: int,
    month: int,
    experiments_dir: Path,
    verbose: bool = True,
    source: str = "binance",
    data_dir: str = "data_raw",
    n_trials: int = 30,
) -> dict:
    """
    Reprocess failed days using optimized params from failed_days study.
    If params don't exist, runs inline optimization first.
    Returns dict with successfully recovered dates and bars DataFrames.
    """
    if not failed_dates:
        return {"recovered_dates": [], "recovered_bars": pd.DataFrame()}

    # Get optimized params for failed days
    params = get_failed_days_optimized_params(experiments_dir, symbol, year, month)

    # If no params exist, run inline optimization
    if params is None:
        if verbose:
            print(f"   ⚠️  No failed_days params found, running inline optimization...")
        params = optimize_failed_days_inline(
            symbol=symbol,
            year=year,
            month=month,
            failed_dates=failed_dates,
            experiments_dir=experiments_dir,
            n_trials=n_trials,
            verbose=verbose,
            source=source,
            data_dir=data_dir,
        )

        if params is None:
            if verbose:
                print(f"   ❌ Inline optimization failed for {year}-{month:02d}")
            return {"recovered_dates": [], "recovered_bars": pd.DataFrame()}

    if verbose:
        print(f"   🔧 Reprocessing {len(failed_dates)} failed days with optimized params:")
        print(f"      λ={params['exp_lambda']:.4f}, T={params['init_exp_T']}")

    recovered_dates = []
    recovered_bars_list = []

    for date_str in failed_dates:
        bars = reprocess_failed_day(symbol, date_str, params, source=source, data_dir=data_dir)

        if bars is not None and not bars.empty:
            recovered_dates.append(date_str)
            recovered_bars_list.append(bars)
            if verbose:
                print(f"      ✅ {date_str}: {len(bars)} bars")
        else:
            if verbose:
                print(f"      ❌ {date_str}: still failing")

    if verbose:
        print(f"   📊 Recovery: {len(recovered_dates)}/{len(failed_dates)} days")

    # Combine all recovered bars
    if recovered_bars_list:
        recovered_bars = pd.concat(recovered_bars_list, ignore_index=True)
    else:
        recovered_bars = pd.DataFrame()

    return {
        "recovered_dates": recovered_dates,
        "recovered_bars": recovered_bars,
        "recovered_bars_list": recovered_bars_list,
    }


def month_range_for_year(year: int) -> list[tuple[int, int]]:
    """Returns list of (year, month) tuples for all 12 months of a year."""
    return [(year, m) for m in range(1, 13)]


def get_date_from_bar(bar_df: pd.DataFrame) -> str:
    """Extract date string (YYYY-MM-DD) from bar DataFrame's open_time."""
    if bar_df is None or bar_df.empty:
        return None
    first_bar_time = bar_df["open_time"].iloc[0]
    if isinstance(first_bar_time, str):
        return first_bar_time[:10]
    elif hasattr(first_bar_time, "strftime"):
        return first_bar_time.strftime("%Y-%m-%d")
    return None


def mark_failed_days(df: pd.DataFrame, failed_dates: set) -> pd.DataFrame:
    """
    Mark bars that belong to failed days.
    Adds columns:
    - failed_day: 1 if the bar's date is in failed_dates, 0 otherwise
    - sample_weight: 1.0 normal, 0.5 for failed
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Extract date from each bar
    df["bar_date"] = df["open_time"].apply(
        lambda x: x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else str(x)[:10]
    )

    # Mark failed days
    df["failed_day"] = df["bar_date"].isin(failed_dates).astype(np.int8)

    # Assign sample weight
    df["sample_weight"] = np.where(df["failed_day"] == 1, SAMPLE_WEIGHT_FAILED, SAMPLE_WEIGHT_NORMAL)

    # Drop temporary column
    df.drop(columns=["bar_date"], inplace=True)

    return df


def _recovered_bar_mask(df: pd.DataFrame) -> pd.Series:
    """Return a safe boolean mask for mixed normal/recovered bar frames."""
    if "_recovered_bar" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["_recovered_bar"].astype("boolean").fillna(False).astype(bool)


def _merge_recovered_day_bars(
    base_bars: pd.DataFrame,
    recovered_bars: pd.DataFrame,
    recovered_dates: set[str],
) -> pd.DataFrame:
    """Replace only recovered days while retaining all other base-day bars."""
    base = base_bars.copy() if base_bars is not None else pd.DataFrame()
    recovered = recovered_bars.copy() if recovered_bars is not None else pd.DataFrame()
    if recovered.empty:
        return base
    day_audits = list(base.attrs.get("day_audits", []))

    if "_bar_date" in base.columns:
        base_dates = base["_bar_date"].astype(str)
    else:
        base_dates = pd.to_datetime(base["open_time"]).dt.strftime("%Y-%m-%d")
    if "_bar_date" in recovered.columns:
        recovered_dates_from_bars = recovered["_bar_date"].astype(str)
    else:
        recovered_dates_from_bars = pd.to_datetime(recovered["open_time"]).dt.strftime("%Y-%m-%d")

    base = base[~base_dates.isin(recovered_dates)]
    recovered["_bar_date"] = recovered_dates_from_bars.to_numpy()
    recovered["_recovered_bar"] = True
    merged = pd.concat([base, recovered], ignore_index=True)
    merged.attrs["day_audits"] = day_audits
    return merged


def _process_month_bars_worker(
    args: tuple,
) -> tuple | None:
    """
    Worker function for parallel month processing — STATELESS.

    Builds ONLY the DRB bars for a month (no features). Feature computation
    runs sequentially in ``build_year_training_data`` through a ``FeatureContext``
    so rolling windows and winsorizer bounds stay causal across months.
    Returns (month, bars_df) or None if the month failed.
    """
    (
        symbol,
        year,
        month,
        params,
        study_name,
        completion_rate,
        min_days,
        source,
        data_dir,
        positioning,
        positioning_data_dir,
    ) = args

    bars_df = build_monthly_daily_bars(
        symbol=symbol,
        year=year,
        month=month,
        params=params,
        min_days=min_days,
        verbose=False,
        source=source,
        data_dir=data_dir,
    )

    if bars_df is None or bars_df.empty:
        return None

    return (month, bars_df)


def build_year_training_data(
    symbol: str,
    year: int,
    output_dir: Path,
    experiments_dir: Path,
    min_days: int = 15,
    dry_run: bool = False,
    verbose: bool = True,
    chunk_size: int = 2,
    recover_failed_days: bool = True,
    source: str = "binance",
    data_dir: str = "data_raw",
    n_trials: int = 30,
    positioning: bool = True,
    positioning_data_dir: str = "data_raw/futures",
    months: list[int] | None = None,
    max_recovery_pct: float = 50.0,
    min_mean_bars_per_day: float = 10.0,
    min_quality_score: float = 20.0,
    max_mad_removed_pct: float = 5.0,
) -> dict:
    """
    Build training data for a single symbol and year.
    Uses chunk-based processing to limit RAM usage.

    Parameters
    ----------
    chunk_size : int
        Number of months to process in memory before saving (default: 2)
        With 2 months, RAM usage is ~14GB max. Use 1 for lower memory.
    recover_failed_days : bool
        If True, reprocess failed days using optimized params (default: True)
    source : str
        Data source: "binance" (download) or "local" (parquet files)
    data_dir : str
        Directory where raw tick parquet files are stored (for --source local)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = HyperparamLoader(experiments_dir)
    positioner = PositioningCalculator(data_dir=positioning_data_dir) if positioning else None

    # Create year directory
    year_dir = output_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    all_failed_dates = set()
    failed_dates_by_month = {}  # {month: [dates]}
    total_bars = 0
    total_failed = 0

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  📈 {symbol} — Year {year} (chunk_size={chunk_size} months)")
        print(f"  Recover failed days: {recover_failed_days}")
        print("=" * 70)

    # Pre-collect hyperparams and failed days for all months
    month_tasks = {}
    requested_months = sorted(set(months or range(1, 13)))
    invalid_months = [m for m in requested_months if m < 1 or m > 12]
    if invalid_months:
        raise ValueError(f"Invalid months: {invalid_months}")

    for month in requested_months:
        hypermeta = loader.get_params_with_meta(symbol, year, month, verbose=False)

        study_gate = validate_study_metadata(
            completion_rate=hypermeta.completion_rate,
            failed_days=hypermeta.failed_days,
        )
        if not study_gate["valid"] and not recover_failed_days:
            raise RuntimeError(
                f"{symbol} {year}-{month:02d}: study rejected: "
                + "; ".join(study_gate["errors"])
            )
        if not study_gate["valid"] and verbose:
            print(
                f"   ⚠️  [{symbol}] {year}-{month:02d}: study has quality warnings: "
                + "; ".join(study_gate["errors"])
            )

        year_failed_dates = {
            fd for fd in hypermeta.failed_days
            if fd.startswith(f"{year}-")
        }
        all_failed_dates.update(year_failed_dates)

        for fd in year_failed_dates:
            fd_month = int(fd.split('-')[1])
            if fd_month not in failed_dates_by_month:
                failed_dates_by_month[fd_month] = []
            if fd not in failed_dates_by_month[fd_month]:
                failed_dates_by_month[fd_month].append(fd)

        month_tasks[month] = (
            symbol,
            year,
            month,
            {"exp_lambda": hypermeta.exp_lambda, "init_exp_T": hypermeta.init_exp_T},
            hypermeta.study_name,
            hypermeta.completion_rate,
            min_days,
            source,
            data_dir,
            positioning,
            positioning_data_dir,
        )

    # Determine parallelism level (cap at 4 to avoid RAM exhaustion)
    n_workers = min(os.cpu_count() or 1, 4)

    all_month_bars = {}

    # Process in chunks of chunk_size months
    for chunk_offset in range(0, len(requested_months), chunk_size):
        chunk_months = requested_months[chunk_offset:chunk_offset + chunk_size]
        chunk_start = chunk_months[0]
        chunk_end = chunk_months[-1]

        if verbose:
            print(f"\n  📦 Processing months {chunk_start}-{chunk_end}...")

        chunk_tasks = [month_tasks[m] for m in chunk_months]
        chunk_bars = {}

        if n_workers > 1 and len(chunk_tasks) > 1:
            # Parallel DRB bar generation within chunk (stateless).
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_process_month_bars_worker, task): task[2] for task in chunk_tasks}
                for future in as_completed(futures):
                    month = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            _, bars_df = result
                            chunk_bars[month] = bars_df
                            if verbose:
                                print(f"   ✅ [{symbol}] {year}-{month:02d}: {len(bars_df):,} DRBs")
                        else:
                            if verbose:
                                print(f"   ⏭️  [{symbol}] {year}-{month:02d}: skipped (no bars)")
                    except Exception as e:
                        if verbose:
                            print(f"   ❌ [{symbol}] {year}-{month:02d}: Error: {e}")
        else:
            # Sequential fallback
            for task in chunk_tasks:
                month = task[2]
                result = _process_month_bars_worker(task)
                if result is not None:
                    _, bars_df = result
                    chunk_bars[month] = bars_df
                    if verbose:
                        print(f"   ✅ [{symbol}] {year}-{month:02d}: {len(bars_df):,} DRBs")
                else:
                    if verbose:
                        print(f"   ⏭️  [{symbol}] {year}-{month:02d}: skipped (no bars)")

        for month in chunk_months:
            bars_df = chunk_bars.get(month)
            if bars_df is not None and not bars_df.empty:
                all_month_bars[month] = bars_df.copy()

    if not all_month_bars:
        if verbose:
            print(f"\n⚠️  [{symbol}] {year}: No data generated")
        return None

    # The generated bars are the source of truth for day-level failure. Study
    # metadata can be stale or sampled from a different data snapshot, so add
    # any base days that actually produced fewer than the minimum bars.
    for month, bars_df in all_month_bars.items():
        if "_base_failed_day" not in bars_df.columns or "_bar_date" not in bars_df.columns:
            continue
        observed_failed = set(
            bars_df.loc[bars_df["_base_failed_day"].astype(bool), "_bar_date"].astype(str)
        )
        if observed_failed:
            failed_dates_by_month.setdefault(month, [])
            failed_dates_by_month[month] = sorted(
                set(failed_dates_by_month[month]) | observed_failed
            )
            all_failed_dates.update(observed_failed)

    daily_audits = []
    for bars_df in all_month_bars.values():
        daily_audits.extend(bars_df.attrs.get("day_audits", []))
    daily_audit = daily_audit_frame(daily_audits)

    # Recover failed days if requested
    still_failed_dates = set()
    recovered_dates_by_month = {}
    recovered_params_by_month = {}

    if recover_failed_days and failed_dates_by_month:
        if verbose:
            print(f"\n  🔧 Recovering failed days...")

        for month, failed_dates in failed_dates_by_month.items():
            if not failed_dates:
                continue

            recovery_result = reprocess_failed_days(
                symbol=symbol,
                failed_dates=failed_dates,
                year=year,
                month=month,
                experiments_dir=experiments_dir,
                verbose=verbose,
                source=source,
                data_dir=data_dir,
                n_trials=n_trials,
            )

            recovered_dates = set(recovery_result["recovered_dates"])
            still_failed = set(failed_dates) - recovered_dates
            still_failed_dates.update(still_failed)

            if not recovery_result["recovered_bars"].empty:
                # Keep recovered DRBs raw until the chronological feature pass.
                # Computing them here would use winsorizer state that already
                # contains later months and would introduce lookahead.
                recovered = recovery_result["recovered_bars"].copy()
                recovered_dates_by_month[month] = recovered_dates
                recovered_params_by_month[month] = (
                    get_failed_days_optimized_params(experiments_dir, symbol, year, month)
                    or {"exp_lambda": 0.9986, "init_exp_T": 5000}
                )
                all_month_bars[month] = _merge_recovered_day_bars(
                    all_month_bars.get(month, pd.DataFrame()),
                    recovered,
                    recovered_dates,
                )
                if verbose:
                    print(f"   📊 Queued {len(recovered):,} recovered DRBs from {len(recovered_dates)} days")

        recovered_frames = []
    else:
        recovered_frames = []
        still_failed_dates = all_failed_dates

    # Recompute all features after recovery bars have been merged into their
    # original months. This keeps rolling features and winsorization causal.
    from src.pipeline.feature_context import FeatureContext

    ctx = FeatureContext(symbol=symbol)
    # Prime the first month with strictly-prior year history. Without this,
    # recovered bars at the beginning of a year are discarded by rolling
    # warm-up before their recovery metadata can reach the output.
    prior_path = output_dir / str(year - 1) / f"{symbol}_{year - 1}.parquet"
    if prior_path.exists():
        try:
            prior = pd.read_parquet(prior_path)
            ctx.seed_bars(prior)
            ctx.seed_winsorizer(prior)
            if verbose:
                print(f"   📚 Seeded causal context from {prior_path.name}")
        except Exception as exc:
            if verbose:
                print(f"   ⚠️  Could not seed prior-year context: {exc}")
    all_feature_frames = []
    for month in sorted(all_month_bars):
        bars_df = all_month_bars[month].copy()
        recovered_dates = recovered_dates_by_month.get(month, set())
        if recovered_dates:
            bar_dates = pd.to_datetime(bars_df["open_time"]).dt.strftime("%Y-%m-%d")
            recovered_flag = _recovered_bar_mask(bars_df)
            # Recovered days replace only their own base-day bars. Normal days
            # remain in the stream, even when another day in the month failed.
            original = bars_df[~bar_dates.isin(recovered_dates) & ~recovered_flag]
            recovered = bars_df[recovered_flag]
            bars_df = pd.concat([original, recovered], ignore_index=True)
        bars_df = bars_df.sort_values("open_time").reset_index(drop=True)

        kwargs = dict(positioning_data_dir=positioning_data_dir)
        if positioner is not None:
            try:
                kwargs["positioning_data"] = positioner.download_positioning_data(symbol, year, month)
            except Exception:
                pass
        bars_features = ctx.compute_features(bars_df, **kwargs)
        if bars_features is None or bars_features.empty:
            continue

        if "_recovered_bar" in bars_features:
            recovered_mask = bars_features.pop("_recovered_bar").astype(bool)
        else:
            recovered_mask = pd.Series(False, index=bars_features.index)
        hypermeta = loader.get_params_with_meta(symbol, year, month, verbose=False)
        bars_features["symbol"] = symbol
        bars_features["year"] = np.int16(year)
        bars_features["month"] = np.int8(month)
        bars_features["exp_lambda"] = np.where(
            recovered_mask,
            recovered_params_by_month.get(month, {}).get("exp_lambda", hypermeta.exp_lambda),
            hypermeta.exp_lambda,
        ).astype(np.float32)
        bars_features["init_exp_T"] = np.where(
            recovered_mask,
            recovered_params_by_month.get(month, {}).get("init_exp_T", hypermeta.init_exp_T),
            hypermeta.init_exp_T,
        ).astype(np.int32)
        bars_features["study_source"] = np.where(
            recovered_mask, f"failed_days_{year}_{month:02d}", hypermeta.study_name
        )
        bars_features["completion_rate"] = np.where(
            recovered_mask, 1.0, hypermeta.completion_rate
        ).astype(np.float32)
        bars_features["recovered_day"] = recovered_mask.astype(np.int8).to_numpy()
        bars_features = bars_features.drop(
            columns=["_bar_date", "_base_failed_day"], errors="ignore"
        )
        all_feature_frames.append(bars_features)

    if not all_feature_frames:
        raise RuntimeError(f"{symbol} {year}: no features generated after recovery merge")

    # Merge feature frames and recovered bars into the final parquet
    if verbose:
        print(f"\n  🔗 Merging monthly features and recovered bars...")

    full_df = pd.concat(all_feature_frames, ignore_index=True)
    del all_feature_frames
    gc.collect()

    # Merge recovered bars if any
    if recovered_frames:
        recovered_df = pd.concat(recovered_frames, ignore_index=True)
        full_df = pd.concat([full_df, recovered_df], ignore_index=True)

        if verbose:
            print(f"   📈 Added {len(recovered_df):,} recovered bars")

        del recovered_df, recovered_frames
        gc.collect()

    # Ensure chronological order — recovered bars from earlier months may be
    # appended at the end, so we sort by open_time before marking failed days.
    full_df = full_df.sort_values("open_time").reset_index(drop=True)

    # Mark failed days (only still-failed ones)
    final_failed_dates = still_failed_dates
    if final_failed_dates:
        full_df = mark_failed_days(full_df, final_failed_dates)
    else:
        full_df["failed_day"] = np.int8(0)
        full_df["sample_weight"] = np.float32(SAMPLE_WEIGHT_NORMAL)

    recovered_day = full_df.get("recovered_day", pd.Series(0, index=full_df.index))
    recovered_day = recovered_day.astype("boolean").fillna(False).astype(np.int8)
    full_df["recovered_day"] = recovered_day
    full_df["recovery_status"] = np.select(
        [
            full_df["recovered_day"].eq(1),
            full_df["failed_day"].eq(1),
        ],
        ["recovered", "failed_irrecoverable"],
        default="normal",
    )
    full_df["partial_month"] = [
        is_partial_month(year, int(month)) for month in full_df["month"]
    ]

    quality_report = monthly_quality_report(full_df)
    quality_gate = validate_monthly_quality(
        quality_report,
        max_recovery_pct=max_recovery_pct,
        min_mean_bars_per_day=min_mean_bars_per_day,
        min_quality_score=min_quality_score,
        allow_partial=True,
    )
    is_full_year = requested_months == list(range(1, 13))
    month_label = "" if is_full_year else "_months_" + "_".join(
        f"{month:02d}" for month in requested_months
    )
    report_path = year_dir / f"{symbol}_{year}{month_label}_quality_report.parquet"
    quality_report.to_parquet(report_path, index=False, engine="pyarrow", compression="snappy")
    daily_report_path = year_dir / f"{symbol}_{year}{month_label}_daily_audit.parquet"
    daily_audit.to_parquet(daily_report_path, index=False, engine="pyarrow", compression="snappy")
    daily_gate = validate_daily_audit(
        daily_audit, max_mad_removed_pct=max_mad_removed_pct
    )
    if not quality_gate["valid"]:
        raise RuntimeError(
            f"{symbol} {year}: monthly quality gate rejected dataset: "
            + "; ".join(quality_gate["errors"])
        )
    if not daily_gate["valid"]:
        raise RuntimeError(
            f"{symbol} {year}: daily audit gate rejected dataset: "
            + "; ".join(daily_gate["errors"])
        )

    full_df["symbol"] = full_df["symbol"].astype("category")
    full_df["study_source"] = full_df["study_source"].astype("category")

    # Save final parquet
    # Never let a month-limited run replace the annual training dataset.
    out_path = year_dir / f"{symbol}_{year}{month_label}.parquet"

    # Calculate total bars AFTER adding recovered bars
    total_bars = len(full_df)

    validation = write_validated_parquet(
        full_df,
        out_path,
        symbol=symbol,
        year=year,
        min_bars=100,
        allow_partial=True,
    )

    # Calculate statistics
    file_size_mb = out_path.stat().st_size / (1024 ** 2)
    failed_bars = full_df["failed_day"].sum()
    failed_pct = failed_bars / total_bars * 100 if total_bars > 0 else 0
    nan_count = full_df.isna().sum().sum()
    nan_pct = nan_count / full_df.size * 100

    if verbose:
        print(f"\n   💾 Final saved → {out_path}")
        print(f"        Bars      : {total_bars:,}")
        print(f"        Failed    : {int(failed_bars):,} ({failed_pct:.1f}%)")
        print(f"        NaN %     : {nan_pct:.2f}%")
        print(f"        Size      : {file_size_mb:.1f} MB")
        print(f"        Features  : validated ({len(validation['feature_unique'])} columns)")

    del full_df
    gc.collect()

    return {
        "symbol": symbol,
        "year": year,
        "total_bars": total_bars,
        "failed_bars": int(failed_bars),
        "failed_pct": failed_pct,
        "file_size_mb": file_size_mb,
        "quality_report": str(report_path),
        "daily_audit": str(daily_report_path),
        "quality_gate": quality_gate,
        "daily_gate": daily_gate,
    }


def print_summary_table(summaries: list[dict]):
    """Print a summary table of all processed symbol-years."""
    if not summaries:
        return

    print("\n" + "=" * 100)
    print("📊 PIPELINE SUMMARY")
    print("=" * 100)
    header = f"{'Symbol':<10} {'Year':>6} {'Bars':>10} {'Failed':>8} {'Failed%':>8} {'Size':>10}"
    print(header)
    print("-" * 100)

    for s in summaries:
        print(
            f"{s['symbol']:<10} "
            f"{s['year']:>6} "
            f"{s['total_bars']:>10,} "
            f"{s['failed_bars']:>8,} "
            f"{s['failed_pct']:>7.1f}% "
            f"{s['file_size_mb']:>9.1f} MB"
        )

    # Totals
    total_bars = sum(s['total_bars'] for s in summaries)
    total_failed = sum(s['failed_bars'] for s in summaries)
    total_size = sum(s['file_size_mb'] for s in summaries)
    print("-" * 100)
    print(f"{'TOTAL':<10} {'':<6} {total_bars:>10,} {total_failed:>8,} {total_failed/max(total_bars,1)*100:>7.1f}% {total_size:>9.1f} MB")
    print("=" * 100)


def append_yesterday(
    symbol: str,
    output_dir: Path,
    experiments_dir: Path,
    source: str = "local",
    data_dir: str = "data_raw",
    positioning: bool = False,
    positioning_data_dir: str = "data_raw/futures",
    verbose: bool = True,
    target_date: str | None = None,
) -> bool:
    """
    Append one day's data to the existing yearly training parquet.

    Fast incremental update (~30s) instead of full year rebuild (~5min).
    Downloads + cleans + generates DRBs for the target day only.
    Loads context from existing yearly parquet for feature computation.

    Args:
        target_date: Specific date to process (YYYY-MM-DD). Default: yesterday.

    Returns True if data was appended, False if skipped (weekend/holiday).
    """
    if target_date:
        yesterday = date.fromisoformat(target_date)
    else:
        yesterday = date.today() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    year = yesterday.year
    month = yesterday.month

    year_parquet = output_dir / str(year) / f"{symbol}_{year}.parquet"
    CONTEXT_BARS = 200  # enough for all rolling window features

    if verbose:
        print(f"\n📅 APPEND-YESTERDAY: {symbol} — {yesterday_str}")
        print(f"   Source: {source} | Output: {year_parquet}")

    # 1. Download + clean ticks for yesterday
    downloader = DownloadData(symbol=symbol, source=source, data_dir=data_dir)
    df_raw = downloader.download_day(yesterday_str)

    if df_raw is None or df_raw.empty:
        if verbose:
            print(f"   ⏭️  No data for {yesterday_str} (weekend/holiday) → skipping")
        return False

    normalizer = DataNormalizer()
    needed = ["timestamp", "price", "quantity"]
    if "dollar_value" in df_raw.columns:
        needed.append("dollar_value")
    df_raw = df_raw[needed].copy()
    df_raw = normalizer.clean_raw_ticks(df_raw, mad_window=100, k=10.0)

    if df_raw is None or df_raw.empty:
        if verbose:
            print(f"   ⚠️  All ticks removed by outlier filter for {yesterday_str}")
        return False

    if "dollar_value" not in df_raw.columns:
        df_raw["dollar_value"] = df_raw["price"] * df_raw["quantity"]

    if verbose:
        print(f"   📥 {len(df_raw):,} ticks downloaded + cleaned")

    # 2. Load hyperparams for yesterday's month
    loader = HyperparamLoader(experiments_dir)
    params = loader.get_params(symbol, year, month, verbose=verbose)

    # 3. Generate DRBs
    bars_gen = OptimizedInfoRunBars(save_path=str(output_dir / "tmp"))
    new_bars = bars_gen.get_drbs(
        df=df_raw,
        exp_lambda=params["exp_lambda"],
        init_exp_T=params["init_exp_T"],
    )

    if new_bars is None or new_bars.empty:
        if verbose:
            print(f"   ⚠️  No bars generated for {yesterday_str}")
        return False

    # Timestamp conversion
    for col in ["open_time", "close_time"]:
        if col in new_bars.columns and new_bars[col].dtype != "datetime64[ns]":
            new_bars[col] = pd.to_datetime(new_bars[col])

    # Downcast
    for col in ["open", "high", "low", "close", "volume", "dollar_value"]:
        if col in new_bars.columns:
            new_bars[col] = new_bars[col].astype(np.float32)
    if "n_ticks" in new_bars.columns:
        new_bars["n_ticks"] = new_bars["n_ticks"].astype(np.int32)

    if verbose:
        print(f"   📊 {len(new_bars)} DRBs generated")

    # 4. Load context from existing yearly parquet (bars BEFORE yesterday only)
    ohclv_cols = ["open_time", "close_time", "open", "high", "low", "close",
                   "n_ticks", "volume", "dollar_value"]

    if year_parquet.exists():
        existing = pd.read_parquet(year_parquet)
        # Filter to bars strictly before yesterday
        yesterday_dt = pd.Timestamp(yesterday_str)
        before_yesterday = existing[pd.to_datetime(existing["open_time"]) < yesterday_dt]
        if len(before_yesterday) == 0:
            before_yesterday = existing.head(1)  # at least one bar for feature warmup
        context = before_yesterday[ohclv_cols].tail(CONTEXT_BARS).copy()
        if verbose:
            last_date = pd.to_datetime(context["open_time"].iloc[-1]).strftime("%Y-%m-%d") if len(context) > 0 else "N/A"
            print(f"   📚 Loaded {len(context)} context bars (before {yesterday_str}, last: {last_date})")
    else:
        context = pd.DataFrame(columns=ohclv_cols)
        if verbose:
            print(f"   🆕 No existing parquet — creating from scratch")

    # 5. Combine context + new bars for feature computation
    combined = pd.concat([context, new_bars], ignore_index=True)
    combined = combined.sort_values("open_time").reset_index(drop=True)

    # 6. Compute features
    from src.features.base_features import FEATURE_COLUMNS
    from src.normalizers.data_normalizer import Winsorizer
    try:
        # Causal winsorization: compute raw features, fit Winsorizer on the
        # historical context bars only, then transform the full frame.
        kwargs = dict(drop_warmup=True, winsorize=False)
        if positioning:
            try:
                positioner = PositioningCalculator(data_dir=positioning_data_dir)
                pos_data = positioner.download_positioning_data(symbol, year, month)
                kwargs["positioning_data"] = pos_data
                kwargs["symbol"] = symbol
                kwargs["positioning_data_dir"] = positioning_data_dir
            except Exception:
                pass

        features = compute_all_features(combined, **kwargs)
        if features is not None and not features.empty:
            context_count = max(len(features) - len(new_bars), 0)
            feat_present = [c for c in FEATURE_COLUMNS if c in features.columns]
            if context_count > 0 and feat_present:
                wz = Winsorizer(limits=(0.01, 0.01)).fit(features.iloc[:context_count][feat_present])
                features = wz.transform(features)
    except Exception as exc:
        if verbose:
            print(f"   ❌ Feature computation failed: {exc}")
        return False

    if features is None or features.empty:
        if verbose:
            print(f"   ❌ No features generated")
        return False

    # 7. Extract only yesterday's rows (last len(new_bars) rows after sorting)
    new_features = features.tail(len(new_bars)).copy()

    if new_features.empty:
        if verbose:
            print(f"   ⚠️  Yesterday's bars dropped during feature warmup")
        return False

    # 8. Add metadata columns
    new_features["symbol"] = symbol
    new_features["year"] = np.int16(year)
    new_features["month"] = np.int8(month)
    new_features["exp_lambda"] = np.float32(params["exp_lambda"])
    new_features["init_exp_T"] = np.int32(params["init_exp_T"])
    new_features["study_source"] = params.get("study_name", "append_yesterday")
    new_features["completion_rate"] = np.float32(1.0)
    new_features["failed_day"] = np.int8(0)
    new_features["sample_weight"] = np.float32(1.0)

    if verbose:
        print(f"   ✅ {len(new_features)} bars with features")

    # 9. Append to yearly parquet
    year_parquet.parent.mkdir(parents=True, exist_ok=True)

    if year_parquet.exists():
        existing = pd.read_parquet(year_parquet)
        # Check if yesterday already exists — skip if so
        existing_dates = pd.to_datetime(existing["open_time"]).dt.date
        yesterday_date = yesterday
        if yesterday_date in existing_dates.values:
            if verbose:
                print(f"   ⏭️  {yesterday_str} already in parquet — skipping")
            return True
        # Simple concat — no dedup needed since we verified date is new
        full = pd.concat([existing, new_features], ignore_index=True)
        full = full.sort_values("open_time").reset_index(drop=True)
    else:
        full = new_features

    full.to_parquet(year_parquet, index=False, engine="pyarrow", compression="snappy")
    sz_mb = year_parquet.stat().st_size / 1e6

    if verbose:
        print(f"   💾 Saved: {year_parquet} ({len(full):,} bars, {sz_mb:.1f} MB)")
        print(f"   ✅ Append complete for {yesterday_str}")

    return True


def append_yesterday_with_recovery(
    symbol: str,
    output_dir: Path,
    experiments_dir: Path,
    source: str = "local",
    data_dir: str = "data_raw",
    positioning: bool = False,
    positioning_data_dir: str = "data_raw/futures",
    verbose: bool = True,
    target_date: str | None = None,
    failed_days_buffer: dict | None = None,
) -> dict:
    """
    Append one day's data with failed-day recovery support.

    Returns dict with keys:
        - status: "appended" | "skipped" | "buffered" | "failed"
        - bars_count: number of bars appended (0 if buffered/failed)
        - date_str: the date processed
    """
    from dataclasses import dataclass

    @dataclass
    class AppendResult:
        status: str
        bars_count: int
        date_str: str

    if target_date:
        target_dt = date.fromisoformat(target_date)
    else:
        target_dt = date.today() - timedelta(days=1)
    target_str = target_dt.strftime("%Y-%m-%d")
    year = target_dt.year
    month = target_dt.month

    year_parquet = output_dir / str(year) / f"{symbol}_{year}.parquet"
    CONTEXT_BARS = 200

    if verbose:
        print(f"\n📅 APPEND-YESTERDAY: {symbol} — {target_str}")
        print(f"   Source: {source} | Output: {year_parquet}")

    downloader = DownloadData(symbol=symbol, source=source, data_dir=data_dir)
    df_raw = downloader.download_day(target_str)

    if df_raw is None or df_raw.empty:
        if verbose:
            print(f"   ⏭️  No data for {target_str} (weekend/holiday) → skipping")
        return {"status": "skipped", "bars_count": 0, "date_str": target_str}

    normalizer = DataNormalizer()
    needed = ["timestamp", "price", "quantity"]
    if "dollar_value" in df_raw.columns:
        needed.append("dollar_value")
    df_raw = df_raw[needed].copy()
    df_raw = normalizer.clean_raw_ticks(df_raw, mad_window=100, k=10.0)

    if df_raw is None or df_raw.empty:
        if verbose:
            print(f"   ⚠️  All ticks removed by outlier filter for {target_str}")
        return {"status": "skipped", "bars_count": 0, "date_str": target_str}

    if "dollar_value" not in df_raw.columns:
        df_raw["dollar_value"] = df_raw["price"] * df_raw["quantity"]

    if verbose:
        print(f"   📥 {len(df_raw):,} ticks downloaded + cleaned")

    loader = HyperparamLoader(experiments_dir)
    params = loader.get_params(symbol, year, month, verbose=verbose)

    bars_gen = OptimizedInfoRunBars(save_path=str(output_dir / "tmp"))
    new_bars = bars_gen.get_drbs(
        df=df_raw,
        exp_lambda=params["exp_lambda"],
        init_exp_T=params["init_exp_T"],
    )

    if new_bars is None or new_bars.empty:
        if verbose:
            print(f"   ⚠️  No bars generated for {target_str}")
        return {"status": "failed", "bars_count": 0, "date_str": target_str}

    for col in ["open_time", "close_time"]:
        if col in new_bars.columns and new_bars[col].dtype != "datetime64[ns]":
            new_bars[col] = pd.to_datetime(new_bars[col])

    for col in ["open", "high", "low", "close", "volume", "dollar_value"]:
        if col in new_bars.columns:
            new_bars[col] = new_bars[col].astype(np.float32)
    if "n_ticks" in new_bars.columns:
        new_bars["n_ticks"] = new_bars["n_ticks"].astype(np.int32)

    if verbose:
        print(f"   📊 {len(new_bars)} DRBs generated")

    if len(new_bars) < MIN_BARS_THRESHOLD:
        if verbose:
            print(f"   ⚠️  {len(new_bars)} DRBs < {MIN_BARS_THRESHOLD} threshold → buffering for recovery")
        if failed_days_buffer is not None:
            key = (year, month)
            if key not in failed_days_buffer:
                failed_days_buffer[key] = []
            failed_days_buffer[key].append(target_str)
        return {"status": "buffered", "bars_count": len(new_bars), "date_str": target_str}

    ohclv_cols = ["open_time", "close_time", "open", "high", "low", "close",
                  "n_ticks", "volume", "dollar_value"]

    if year_parquet.exists():
        existing = pd.read_parquet(year_parquet)
        yesterday_dt = pd.Timestamp(target_str)
        before_target = existing[pd.to_datetime(existing["open_time"]) < yesterday_dt]
        if len(before_target) == 0:
            before_target = existing.head(1)
        context = before_target[ohclv_cols].tail(CONTEXT_BARS).copy()
        if verbose:
            last_date = pd.to_datetime(context["open_time"].iloc[-1]).strftime("%Y-%m-%d") if len(context) > 0 else "N/A"
            print(f"   📚 Loaded {len(context)} context bars (before {target_str}, last: {last_date})")
    else:
        context = pd.DataFrame(columns=ohclv_cols)
        if verbose:
            print(f"   🆕 No existing parquet — creating from scratch")

    combined = pd.concat([context, new_bars], ignore_index=True)
    combined = combined.sort_values("open_time").reset_index(drop=True)

    try:
        kwargs = dict(drop_warmup=True, winsorize=True)
        if positioning:
            try:
                positioner = PositioningCalculator(data_dir=positioning_data_dir)
                pos_data = positioner.download_positioning_data(symbol, year, month)
                kwargs["positioning_data"] = pos_data
                kwargs["symbol"] = symbol
                kwargs["positioning_data_dir"] = positioning_data_dir
            except Exception:
                pass

        features = compute_all_features(combined, **kwargs)
    except Exception as exc:
        if verbose:
            print(f"   ❌ Feature computation failed: {exc}")
        return {"status": "failed", "bars_count": 0, "date_str": target_str}

    if features is None or features.empty:
        if verbose:
            print(f"   ❌ No features generated")
        return {"status": "failed", "bars_count": 0, "date_str": target_str}

    new_features = features.tail(len(new_bars)).copy()

    if new_features.empty:
        if verbose:
            print(f"   ⚠️  {target_str}'s bars dropped during feature warmup")
        return {"status": "failed", "bars_count": 0, "date_str": target_str}

    new_features["symbol"] = symbol
    new_features["year"] = np.int16(year)
    new_features["month"] = np.int8(month)
    new_features["exp_lambda"] = np.float32(params["exp_lambda"])
    new_features["init_exp_T"] = np.int32(params["init_exp_T"])
    new_features["study_source"] = params.get("study_name", "append_yesterday")
    new_features["completion_rate"] = np.float32(1.0)
    new_features["failed_day"] = np.int8(0)
    new_features["sample_weight"] = np.float32(1.0)

    if verbose:
        print(f"   ✅ {len(new_features)} bars with features")

    year_parquet.parent.mkdir(parents=True, exist_ok=True)

    if year_parquet.exists():
        existing = pd.read_parquet(year_parquet)
        existing_dates = pd.to_datetime(existing["open_time"]).dt.date
        target_date = target_dt
        if target_date in existing_dates.values:
            if verbose:
                print(f"   ⏭️  {target_str} already in parquet — skipping")
            return {"status": "skipped", "bars_count": 0, "date_str": target_str}
        full = pd.concat([existing, new_features], ignore_index=True)
        full = full.sort_values("open_time").reset_index(drop=True)
    else:
        full = new_features

    full.to_parquet(year_parquet, index=False, engine="pyarrow", compression="snappy")
    sz_mb = year_parquet.stat().st_size / 1e6

    if verbose:
        print(f"   💾 Saved: {year_parquet} ({len(full):,} bars, {sz_mb:.1f} MB)")
        print(f"   ✅ Append complete for {target_str}")

    return {"status": "appended", "bars_count": len(new_features), "date_str": target_str}


def reprocess_single_day_features(
    symbol: str,
    year: int,
    month: int,
    bars_df: pd.DataFrame,
    params: dict,
    positioning: bool,
    positioning_data_dir: str,
) -> pd.DataFrame | None:
    """Compute features for a single day (already has DRBs), used in recovery."""
    if bars_df is None or bars_df.empty:
        return None

    bars_df = bars_df.copy()
    for col in ["open_time", "close_time"]:
        if col in bars_df.columns and bars_df[col].dtype != "datetime64[ns]":
            bars_df[col] = pd.to_datetime(bars_df[col])
    for col in ["open", "high", "low", "close", "volume", "dollar_value"]:
        if col in bars_df.columns:
            bars_df[col] = bars_df[col].astype(np.float32)
    if "n_ticks" in bars_df.columns:
        bars_df["n_ticks"] = bars_df["n_ticks"].astype(np.int32)

    CONTEXT_BARS = 200
    ohclv_cols = ["open_time", "close_time", "open", "high", "low", "close",
                  "n_ticks", "volume", "dollar_value"]
    year_parquet = Path("data_optimized/training") / str(year) / f"{symbol}_{year}.parquet"

    if year_parquet.exists():
        existing = pd.read_parquet(year_parquet)
        first_bar_time = bars_df["open_time"].iloc[0]
        if hasattr(first_bar_time, "strftime"):
            target_dt = first_bar_time
        else:
            target_dt = pd.Timestamp(first_bar_time)
        before_target = existing[pd.to_datetime(existing["open_time"]) < target_dt]
        if len(before_target) == 0:
            before_target = existing.head(1)
        context = before_target[ohclv_cols].tail(CONTEXT_BARS).copy()
    else:
        context = pd.DataFrame(columns=ohclv_cols)

    combined = pd.concat([context, bars_df], ignore_index=True)
    combined = combined.sort_values("open_time").reset_index(drop=True)

    try:
        kwargs = dict(drop_warmup=True, winsorize=True)
        if positioning:
            try:
                positioner = PositioningCalculator(data_dir=positioning_data_dir)
                pos_data = positioner.download_positioning_data(symbol, year, month)
                kwargs["positioning_data"] = pos_data
                kwargs["symbol"] = symbol
                kwargs["positioning_data_dir"] = positioning_data_dir
            except Exception:
                pass

        features = compute_all_features(combined, **kwargs)
    except Exception:
        return None

    if features is None or features.empty:
        return None

    new_features = features.tail(len(bars_df)).copy()
    if new_features.empty:
        return None

    new_features["symbol"] = symbol
    new_features["year"] = np.int16(year)
    new_features["month"] = np.int8(month)
    new_features["exp_lambda"] = np.float32(params["exp_lambda"])
    new_features["init_exp_T"] = np.int32(params["init_exp_T"])
    new_features["study_source"] = f"failed_days_{year}_{month:02d}"
    new_features["completion_rate"] = np.float32(1.0)

    return new_features


def append_range(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    experiments_dir: Path,
    source: str = "local",
    data_dir: str = "data_raw",
    positioning: bool = False,
    positioning_data_dir: str = "data_raw/futures",
    verbose: bool = True,
    n_trials: int = 20,
) -> dict:
    """
    Append data for a range of dates to the existing yearly training parquet.

    1. Process each day using append_yesterday_with_recovery
    2. Buffer days with <40 bars per (year, month)
    3. When a month accumulates >= FAILED_DAYS_MIN_TO_OPTIMIZE failed days,
       run inline optimization + recovery for that month
    4. Append recovered bars with optimized params; still-failed bars
       appended with failed_day=1 and sample_weight=0.5
    """
    from datetime import datetime, timedelta

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        if verbose:
            print(f"❌ append_range: start {start_date} is after end {end_date}")
        return {"processed": 0, "skipped": 0, "buffered": 0, "recovered": 0, "failed": 0}

    current = start
    stats = {"processed": 0, "skipped": 0, "buffered": 0, "recovered": 0, "failed": 0}
    failed_days_buffer: dict = {}

    if verbose:
        print(f"\n📅 APPEND-RANGE: {symbol} — {start_date} → {end_date}")

    while current <= end:
        current_str = current.strftime("%Y-%m-%d")
        result = append_yesterday_with_recovery(
            symbol=symbol,
            output_dir=output_dir,
            experiments_dir=experiments_dir,
            source=source,
            data_dir=data_dir,
            positioning=positioning,
            positioning_data_dir=positioning_data_dir,
            verbose=verbose,
            target_date=current_str,
            failed_days_buffer=failed_days_buffer,
        )

        if result["status"] == "appended":
            stats["processed"] += 1
        elif result["status"] == "skipped":
            stats["skipped"] += 1
        elif result["status"] == "buffered":
            stats["buffered"] += 1
        else:
            stats["failed"] += 1

        current += timedelta(days=1)

    if verbose:
        print(f"\n📦 First pass complete: {stats['processed']} processed, "
              f"{stats['skipped']} skipped, {stats['buffered']} buffered, {stats['failed']} failed")

    if failed_days_buffer:
        if verbose:
            print(f"\n🔧 Starting failed days recovery...")

        for (year, month), failed_dates in sorted(failed_days_buffer.items()):
            if len(failed_dates) < FAILED_DAYS_MIN_TO_OPTIMIZE:
                if verbose:
                    print(f"   ⏭️  {year}-{month:02d}: only {len(failed_dates)} buffered days "
                          f"(< {FAILED_DAYS_MIN_TO_OPTIMIZE}), skipping optimization")
                for date_str in failed_dates:
                    _append_failed_day(
                        symbol=symbol,
                        date_str=date_str,
                        output_dir=output_dir,
                        experiments_dir=experiments_dir,
                        source=source,
                        data_dir=data_dir,
                        positioning=positioning,
                        positioning_data_dir=positioning_data_dir,
                        verbose=verbose,
                    )
                    stats["failed"] += 1
                    stats["buffered"] -= 1
                continue

            if verbose:
                print(f"\n   🔬 Optimizing {len(failed_dates)} failed days for {symbol} {year}-{month:02d}")

            recovery_result = reprocess_failed_days(
                symbol=symbol,
                failed_dates=failed_dates,
                year=year,
                month=month,
                experiments_dir=experiments_dir,
                verbose=verbose,
                source=source,
                data_dir=data_dir,
                n_trials=n_trials,
            )

            recovered_dates = set(recovery_result["recovered_dates"])
            still_failing = set(failed_dates) - recovered_dates

            for date_str in recovered_dates:
                recovered_bars = None
                for rb in recovery_result["recovered_bars_list"]:
                    first_dt = rb["open_time"].iloc[0]
                    dt_str = first_dt.strftime("%Y-%m-%d") if hasattr(first_dt, "strftime") else str(first_dt)[:10]
                    if dt_str == date_str:
                        recovered_bars = rb
                        break

                if recovered_bars is not None and not recovered_bars.empty:
                    failed_params = get_failed_days_optimized_params(experiments_dir, symbol, year, month)
                    if failed_params:
                        feat = reprocess_single_day_features(
                            symbol=symbol,
                            year=year,
                            month=month,
                            bars_df=recovered_bars,
                            params=failed_params,
                            positioning=positioning,
                            positioning_data_dir=positioning_data_dir,
                        )
                        if feat is not None and not feat.empty:
                            feat["failed_day"] = np.int8(0)
                            feat["sample_weight"] = np.float32(1.0)
                            _append_features_to_parquet(
                                symbol=symbol,
                                year=year,
                                feat=feat,
                                output_dir=output_dir,
                                verbose=verbose,
                            )
                            stats["recovered"] += 1
                            stats["buffered"] -= 1
                            if verbose:
                                print(f"      ✅ {date_str}: recovered {len(feat)} bars")
                            continue

                if verbose:
                    print(f"      ❌ {date_str}: recovery produced no features")
                _append_failed_day(
                    symbol=symbol,
                    date_str=date_str,
                    output_dir=output_dir,
                    experiments_dir=experiments_dir,
                    source=source,
                    data_dir=data_dir,
                    positioning=positioning,
                    positioning_data_dir=positioning_data_dir,
                    verbose=verbose,
                )
                stats["failed"] += 1
                stats["buffered"] -= 1

            for date_str in still_failing:
                _append_failed_day(
                    symbol=symbol,
                    date_str=date_str,
                    output_dir=output_dir,
                    experiments_dir=experiments_dir,
                    source=source,
                    data_dir=data_dir,
                    positioning=positioning,
                    positioning_data_dir=positioning_data_dir,
                    verbose=verbose,
                )
                stats["failed"] += 1
                stats["buffered"] -= 1

    if verbose:
        print(f"\n📊 Range stats: {stats['processed']} processed, "
              f"{stats['skipped']} skipped, {stats['recovered']} recovered, {stats['failed']} failed")

    return stats


def _load_existing_dates(
    symbol: str,
    year: int,
    output_dir: Path,
) -> set[date]:
    year_parquet = output_dir / str(year) / f"{symbol}_{year}.parquet"
    if not year_parquet.exists():
        return set()

    try:
        existing = pd.read_parquet(year_parquet, columns=["open_time"])
    except Exception:
        return set()

    return set(pd.to_datetime(existing["open_time"]).dt.date)


def _build_missing_ranges(missing_dates: list[date]) -> list[tuple[date, date]]:
    if not missing_dates:
        return []

    ranges = []
    start = missing_dates[0]
    prev = missing_dates[0]

    for current in missing_dates[1:]:
        if current == prev + timedelta(days=1):
            prev = current
            continue
        ranges.append((start, prev))
        start = current
        prev = current

    ranges.append((start, prev))
    return ranges


def append_missing_range(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    experiments_dir: Path,
    source: str = "auto",
    data_dir: str = "data_raw",
    positioning: bool = False,
    positioning_data_dir: str = "data_raw/futures",
    verbose: bool = True,
    n_trials: int = 20,
) -> dict:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        if verbose:
            print(f"❌ append_missing_range: start {start_date} is after end {end_date}")
        return {"processed": 0, "skipped": 0, "buffered": 0, "recovered": 0, "failed": 0}

    years = range(start.year, end.year + 1)
    existing_by_year = {
        y: _load_existing_dates(symbol, y, output_dir) for y in years
    }

    all_dates = []
    current = start
    while current <= end:
        all_dates.append(current)
        current += timedelta(days=1)

    missing_dates = [d for d in all_dates if d not in existing_by_year.get(d.year, set())]

    if not missing_dates:
        if verbose:
            print(f"   ⏭️  {symbol}: no missing days in {start_date} → {end_date}")
        return {"processed": 0, "skipped": 0, "buffered": 0, "recovered": 0, "failed": 0}

    missing_dates.sort()
    ranges = _build_missing_ranges(missing_dates)

    if verbose:
        print(f"   📌 {symbol}: {len(missing_dates)} missing day(s) in {start_date} → {end_date}")

    totals = {"processed": 0, "skipped": 0, "buffered": 0, "recovered": 0, "failed": 0}

    for r_start, r_end in ranges:
        stats = append_range(
            symbol=symbol,
            start_date=r_start.strftime("%Y-%m-%d"),
            end_date=r_end.strftime("%Y-%m-%d"),
            output_dir=output_dir,
            experiments_dir=experiments_dir,
            source=source,
            data_dir=data_dir,
            positioning=positioning,
            positioning_data_dir=positioning_data_dir,
            verbose=verbose,
            n_trials=n_trials,
        )

        for key in totals:
            totals[key] += stats.get(key, 0)

    return totals


def append_missing_month(
    symbol: str,
    month_str: str,
    output_dir: Path,
    experiments_dir: Path,
    source: str = "auto",
    data_dir: str = "data_raw",
    positioning: bool = False,
    positioning_data_dir: str = "data_raw/futures",
    verbose: bool = True,
    n_trials: int = 20,
) -> dict:
    try:
        year = int(month_str[:4])
        month = int(month_str[5:7])
    except Exception:
        if verbose:
            print(f"❌ append_missing_month: invalid month format {month_str} (use YYYY-MM)")
        return {"processed": 0, "skipped": 0, "buffered": 0, "recovered": 0, "failed": 0}

    start = date(year, month, 1)
    last_day = monthrange(year, month)[1]
    end = date(year, month, last_day)
    end = min(end, date.today() - timedelta(days=1))

    if end < start:
        if verbose:
            print(f"   ⏭️  {symbol}: no closed days for {month_str}")
        return {"processed": 0, "skipped": 0, "buffered": 0, "recovered": 0, "failed": 0}

    return append_missing_range(
        symbol=symbol,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        output_dir=output_dir,
        experiments_dir=experiments_dir,
        source=source,
        data_dir=data_dir,
        positioning=positioning,
        positioning_data_dir=positioning_data_dir,
        verbose=verbose,
        n_trials=n_trials,
    )


def deduplicate_yearly_parquet(
    symbol: str,
    year: int,
    output_dir: Path,
    verbose: bool = True,
) -> dict:
    year_parquet = output_dir / str(year) / f"{symbol}_{year}.parquet"

    if not year_parquet.exists():
        if verbose:
            print(f"   ⏭️  {symbol} {year}: parquet not found")
        return {"removed_true_dups": 0, "removed_ts_dups": 0, "final_rows": 0}

    df = pd.read_parquet(year_parquet)
    if df.empty or "open_time" not in df.columns:
        if verbose:
            print(f"   ⏭️  {symbol} {year}: no data to dedup")
        return {"removed_true_dups": 0, "removed_ts_dups": 0, "final_rows": len(df)}

    before = len(df)
    key_cols = ["open_time", "open", "high", "low", "close", "volume"]
    if all(c in df.columns for c in key_cols):
        df = df.drop_duplicates(subset=key_cols, keep="first")

    after_true = len(df)

    # Prefer the most liquid bar when timestamps collide
    metric_col = "dollar_value" if "dollar_value" in df.columns else "volume"
    df["_open_time_sort"] = pd.to_datetime(df["open_time"], errors="coerce")

    if metric_col in df.columns:
        df = df.sort_values(["_open_time_sort", metric_col], ascending=[True, False])
    else:
        df = df.sort_values(["_open_time_sort"])

    df = df.drop_duplicates(subset=["_open_time_sort"], keep="first")
    df = df.sort_values(["_open_time_sort"]).drop(columns=["_open_time_sort"]).reset_index(drop=True)

    after_ts = len(df)
    removed_true = before - after_true
    removed_ts = after_true - after_ts

    df.to_parquet(year_parquet, index=False, engine="pyarrow", compression="snappy")

    if verbose:
        print(
            f"   ✅ {symbol} {year}: removed {removed_true} true duplicates, "
            f"{removed_ts} timestamp duplicates (final {after_ts:,} rows)"
        )

    return {
        "removed_true_dups": removed_true,
        "removed_ts_dups": removed_ts,
        "final_rows": after_ts,
    }


def _append_failed_day(
    symbol: str,
    date_str: str,
    output_dir: Path,
    experiments_dir: Path,
    source: str = "binance",
    data_dir: str = "data_raw",
    positioning: bool = False,
    positioning_data_dir: str = "data_raw/futures",
    verbose: bool = True,
) -> bool:
    """Append a still-failed day with failed_day=1 and sample_weight=0.0.

    Days that fail even after targeted recovery are structural low-liquidity
    events where DRB bars carry no IID information — excluded from training."""

    from datetime import datetime

    year = int(date_str[:4])
    month = int(date_str[5:7])

    try:
        downloader = DownloadData(symbol=symbol, source=source, data_dir=data_dir)
        normalizer = DataNormalizer()
        bars_gen = OptimizedInfoRunBars(save_path=str(output_dir / "tmp"))

        df_raw = downloader.download_day(date_str)
        if df_raw is None or df_raw.empty:
            return False

        df_clean = normalizer.clean_raw_ticks(df_raw, mad_window=100, k=10.0)
        if df_clean is None or df_clean.empty:
            return False

        if "dollar_value" not in df_clean.columns:
            df_clean["dollar_value"] = df_clean["price"] * df_clean["quantity"]

        loader = HyperparamLoader(experiments_dir)
        params = loader.get_params(symbol, year, month, verbose=False)

        bars = bars_gen.get_drbs(
            df=df_clean,
            exp_lambda=params["exp_lambda"],
            init_exp_T=params["init_exp_T"],
        )

        if bars is None or bars.empty:
            if verbose:
                print(f"      ❌ {date_str}: no bars even with default params")
            return False

        feat = reprocess_single_day_features(
            symbol=symbol,
            year=year,
            month=month,
            bars_df=bars,
            params=params,
            positioning=positioning,
            positioning_data_dir=positioning_data_dir,
        )

        if feat is None or feat.empty:
            if verbose:
                print(f"      ❌ {date_str}: no features generated")
            return False

        feat["failed_day"] = np.int8(1)
        feat["sample_weight"] = np.float32(0.0)

        _append_features_to_parquet(
            symbol=symbol,
            year=year,
            feat=feat,
            output_dir=output_dir,
            verbose=verbose,
        )

        if verbose:
            print(f"      ⚠️  {date_str}: appended as structural failed_day=1 ({len(feat)} bars, weight=0.0)")
        return True
    except Exception as e:
        if verbose:
            print(f"      ❌ {date_str}: error - {e}")
        return False


def _append_features_to_parquet(
    symbol: str,
    year: int,
    feat: pd.DataFrame,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """Append a features DataFrame to the yearly parquet."""
    year_parquet = output_dir / str(year) / f"{symbol}_{year}.parquet"
    year_parquet.parent.mkdir(parents=True, exist_ok=True)

    if year_parquet.exists():
        existing = pd.read_parquet(year_parquet)
        full = pd.concat([existing, feat], ignore_index=True)
        full = full.sort_values("open_time").reset_index(drop=True)
    else:
        full = feat

    full.to_parquet(year_parquet, index=False, engine="pyarrow", compression="snappy")

    if verbose:
        sz_mb = year_parquet.stat().st_size / 1e6
        print(f"      💾 Updated {year_parquet.name} ({len(full):,} total bars, {sz_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Build yearly training data with failed day tracking"
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Single year to process (e.g., 2023)"
    )
    parser.add_argument(
        "--start-year", type=int, default=None,
        help="Start year for range (e.g., 2023)"
    )
    parser.add_argument(
        "--end-year", type=int, default=None,
        help="End year for range (e.g., 2025)"
    )
    parser.add_argument(
        "--symbols", nargs="+",
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
        help="Symbols to process"
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "data_optimized" / "training"),
        help="Output directory for parquet files (default: trading-core/data_optimized/training)"
    )
    parser.add_argument(
        "--experiments-dir",
        default=str(Path(__file__).resolve().parent.parent / "experiments"),
        help="Path to experiments directory (default: trading-core/experiments)"
    )
    parser.add_argument(
        "--min-days", type=int, default=15,
        help="Minimum successful days per month (default: 15)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2,
        help="Months to process in memory before saving (default: 2, ~14GB RAM)"
    )
    parser.add_argument(
        "--months", nargs="+", type=int, default=None,
        help="Optional subset of calendar months to process (e.g. --months 1 2 3)"
    )
    parser.add_argument(
        "--max-recovery-pct", type=float, default=50.0,
        help="Reject non-current months with more recovered bars than this percentage"
    )
    parser.add_argument(
        "--min-mean-bars-per-day", type=float, default=10.0,
        help="Minimum mean bars/day for non-current months"
    )
    parser.add_argument(
        "--min-quality-score", type=float, default=20.0,
        help="Minimum statistical quality score for non-current months"
    )
    parser.add_argument(
        "--max-mad-removed-pct", type=float, default=5.0,
        help="Reject datasets with a day exceeding this MAD removal percentage"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without processing"
    )
    parser.add_argument(
        "--no-recover", action="store_true",
        help="Skip failed day recovery (don't use optimized params)"
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Number of trials for failed days optimization (default: 30)"
    )
    parser.add_argument(
        "--source", type=str, default="auto", choices=["binance", "local", "auto"],
        help="Data source: binance (download), local (parquet files), or auto (local + fallback)"
    )
    parser.add_argument(
        "--data-dir", default="data_raw",
        help="Directory where raw tick parquet files are stored (for --source local)"
    )
    parser.add_argument(
        "--no-positioning",
        action="store_true",
        help="Skip positioning features (funding rate, OI, taker ratio, L/S ratio)",
    )
    parser.add_argument(
        "--positioning-data-dir", default="data_raw/futures",
        help="Directory for caching Binance Futures data (default: data_raw/futures)",
    )
    parser.add_argument(
        "--append-yesterday",
        action="store_true",
        help="Fast mode: process only yesterday and append to existing yearly parquet (~30s)",
    )
    parser.add_argument(
        "--append-date", type=str, default=None,
        help="Specific date to append (YYYY-MM-DD). Use with --append-yesterday.",
    )
    parser.add_argument(
        "--append-range", type=str, nargs=2, metavar=("START", "END"),
        help="Date range to append (YYYY-MM-DD YYYY-MM-DD). Processes each day in range "
             "that is missing from the existing parquet. Uses same walk-forward hyperparams "
             "as the full pipeline.",
    )
    parser.add_argument(
        "--append-missing-range", type=str, nargs=2, metavar=("START", "END"),
        help="Append only missing days in a range (YYYY-MM-DD YYYY-MM-DD). Uses walk-forward hyperparams.",
    )
    parser.add_argument(
        "--append-missing-month", type=str, default=None,
        help="Append only missing days for a month (YYYY-MM). Uses walk-forward hyperparams.",
    )
    parser.add_argument(
        "--cap-today",
        action="store_true",
        help="Cap end date at today-1 to avoid incomplete current-day ticks",
    )
    parser.add_argument(
        "--failed-days-trials", type=int, default=20,
        help="Number of trials for inline optimization of failed days (default: 20)",
    )
    parser.add_argument(
        "--dedup-year", type=int, default=None,
        help="Deduplicate existing yearly parquet for a specific year",
    )

    args = parser.parse_args()

    # Normalize symbols: handle both comma-separated and space-separated
    # (run_year_optimization.py passes comma-separated, this script used nargs="+")
    symbols = []
    for s in args.symbols:
        symbols.extend(s.split(','))
    args.symbols = [s.strip() for s in symbols if s.strip()]

    if args.dedup_year is not None:
        output_dir = Path(args.output_dir)
        print(f"\n🧹 DEDUP mode: year {args.dedup_year}")
        for symbol in args.symbols:
            deduplicate_yearly_parquet(
                symbol=symbol,
                year=args.dedup_year,
                output_dir=output_dir,
                verbose=True,
            )
        print("\n✅ Dedup complete!\n")
        return

    # Fast append mode — no year/symbol range needed
    if args.append_yesterday or args.append_range or args.append_missing_range or args.append_missing_month:
        output_dir = Path(args.output_dir)
        experiments_dir = Path(args.experiments_dir)

        if args.append_missing_month:
            month_str = args.append_missing_month
            print(f"\n📅 APPEND-MISSING-MONTH mode: {month_str}")
            for symbol in args.symbols:
                stats = append_missing_month(
                    symbol=symbol,
                    month_str=month_str,
                    output_dir=output_dir,
                    experiments_dir=experiments_dir,
                    source=args.source,
                    data_dir=args.data_dir,
                    positioning=not args.no_positioning,
                    positioning_data_dir=args.positioning_data_dir,
                    verbose=True,
                    n_trials=args.failed_days_trials,
                )
            print("\n✅ Append-missing-month complete!\n")
            return

        if args.append_missing_range:
            start_date, end_date = args.append_missing_range
            if args.cap_today:
                capped = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
                if end_date > capped:
                    end_date = capped
            print(f"\n📅 APPEND-MISSING-RANGE mode: {start_date} → {end_date}")
            for symbol in args.symbols:
                stats = append_missing_range(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=output_dir,
                    experiments_dir=experiments_dir,
                    source=args.source,
                    data_dir=args.data_dir,
                    positioning=not args.no_positioning,
                    positioning_data_dir=args.positioning_data_dir,
                    verbose=True,
                    n_trials=args.failed_days_trials,
                )
            print("\n✅ Append-missing-range complete!\n")
            return

        if args.append_range:
            start_date, end_date = args.append_range
            if args.cap_today:
                capped = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
                if end_date > capped:
                    end_date = capped
            print(f"\n📅 APPEND-RANGE mode: {start_date} → {end_date}")
            for symbol in args.symbols:
                stats = append_range(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=output_dir,
                    experiments_dir=experiments_dir,
                    source=args.source,
                    data_dir=args.data_dir,
                    positioning=not args.no_positioning,
                    positioning_data_dir=args.positioning_data_dir,
                    verbose=True,
                    n_trials=args.failed_days_trials,
                )
            print("\n✅ Append-range complete!\n")
            return

        for symbol in args.symbols:
            success = append_yesterday(
                symbol=symbol,
                output_dir=output_dir,
                experiments_dir=experiments_dir,
                source=args.source,
                data_dir=args.data_dir,
                positioning=not args.no_positioning,
                positioning_data_dir=args.positioning_data_dir,
                verbose=True,
                target_date=args.append_date,
            )
            if not success:
                print(f"   ⏭️  {symbol}: no data appended (weekend/holiday or no ticks)")
        print("\n✅ Append-yesterday complete!\n")
        return

    # Determine years to process
    if args.year is not None:
        years = [args.year]
    elif args.start_year is not None and args.end_year is not None:
        years = list(range(args.start_year, args.end_year + 1))
    else:
        print("Error: Must specify --year or --start-year/--end-year")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    experiments_dir = Path(args.experiments_dir)

    print("\n" + "🚀" * 20)
    print(f"  YEARLY TRAINING DATA PIPELINE")
    print(f"  Symbols   : {', '.join(args.symbols)}")
    print(f"  Years     : {years[0]} → {years[-1]} ({len(years)} years)")
    print(f"  Chunk size: {args.chunk_size} months")
    print(f"  Recover   : {not args.no_recover}")
    print(f"  Positioning: {'enabled' if not args.no_positioning else 'disabled'}")
    print(f"  Source    : {args.source}")
    print(f"  Output    : {output_dir}")
    print("🚀" * 20 + "\n")

    summaries = []

    for symbol in args.symbols:
        for year in years:
            if args.dry_run:
                loader = HyperparamLoader(experiments_dir)
                print(f"\n🔍 DRY RUN: {symbol} - {year}")
                for month in sorted(set(args.months or range(1, 13))):
                    hypermeta = loader.get_params_with_meta(symbol, year, month, verbose=True)
                continue

            result = build_year_training_data(
                symbol=symbol,
                year=year,
                output_dir=output_dir,
                experiments_dir=experiments_dir,
                min_days=args.min_days,
                dry_run=args.dry_run,
                verbose=True,
                chunk_size=args.chunk_size,
                recover_failed_days=not args.no_recover,
                source=args.source,
                data_dir=args.data_dir,
                n_trials=args.trials,
                positioning=not args.no_positioning,
                positioning_data_dir=args.positioning_data_dir,
                months=args.months,
                max_recovery_pct=args.max_recovery_pct,
                min_mean_bars_per_day=args.min_mean_bars_per_day,
                min_quality_score=args.min_quality_score,
                max_mad_removed_pct=args.max_mad_removed_pct,
            )

            if result:
                summaries.append(result)

    if not args.dry_run:
        print_summary_table(summaries)
        print("\n✅ Pipeline complete!\n")


if __name__ == "__main__":
    main()
