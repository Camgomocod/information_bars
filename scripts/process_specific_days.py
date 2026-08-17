#!/usr/bin/env python3
"""
process_specific_days.py — Genera DRB training data para días específicos

Flujo por cada día:
1. Carga hyperparams del estudio walk-forward anterior
2. Carga y limpia ticks crudos
3. Genera DRBs
4. Si las barras son insuficientes → recuperación automática (optimización inline)
5. Calcula features (VectorBT + positioning)
6. Marca failed_days y guarda parquet

Usage:
    # Día específico
    micromamba run -n trading-core python scripts/process_specific_days.py \
        --symbol BTCUSDT --dates 2026-01-15

    # Múltiples días y símbolos
    micromamba run -n trading-core python scripts/process_specific_days.py \
        --symbols BTCUSDT ETHUSDT --dates 2026-01-15 2026-01-20

    # Sin positioning features
    micromamba run -n trading-core python scripts/process_specific_days.py \
        --symbol BTCUSDT --dates 2026-01-15 --no-positioning

    # Usar datos locales (sin descargar)
    micromamba run -n trading-core python scripts/process_specific_days.py \
        --symbol BTCUSDT --dates 2026-01-15 --source local --data-dir data_raw
"""

import argparse
import gc
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Reuse failed-day recovery logic from build_yearly_training_data
from scripts.build_yearly_training_data import (
    MIN_BARS_THRESHOLD,
    SAMPLE_WEIGHT_NORMAL,
    mark_failed_days,
    reprocess_failed_days,
)
from src.bars.info_bars import OptimizedInfoRunBars
from src.connectors.download_data import DownloadData
from src.features.positioning_calculator import PositioningCalculator
from src.normalizers.data_normalizer import DataNormalizer
from src.pipeline.hyperparam_loader import HyperparamLoader

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]


def parse_date(date_str: str) -> tuple[int, int, int]:
    parts = date_str.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")
    return int(parts[0]), int(parts[1]), int(parts[2])


def process_single_day(
    symbol: str,
    date_str: str,
    exp_lambda: float,
    init_exp_T: int,
    source: str,
    data_dir: str,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """Load ticks for one day, clean, generate DRBs with given params."""
    try:
        downloader = DownloadData(symbol=symbol, source=source, data_dir=data_dir)
        normalizer = DataNormalizer()
        bars_gen = OptimizedInfoRunBars(save_path=str(project_root / "data_optimized" / "tmp"))

        df_raw = downloader.download_day(date_str)
        if df_raw is None or df_raw.empty:
            if verbose:
                print(f"   ❌ {symbol} {date_str}: No data")
            return None

        needed = ["timestamp", "price", "quantity"]
        if "dollar_value" in df_raw.columns:
            needed.append("dollar_value")
        df_raw = df_raw[needed].copy()

        df_clean = normalizer.clean_raw_ticks(df_raw, mad_window=100, k=10.0)
        if df_clean is None or df_clean.empty:
            if verbose:
                print(f"   ❌ {symbol} {date_str}: All ticks removed by outlier filter")
            return None

        if "dollar_value" not in df_clean.columns:
            df_clean["dollar_value"] = df_clean["price"] * df_clean["quantity"]

        bars = bars_gen.get_drbs(
            df_clean,
            exp_lambda=exp_lambda,
            init_exp_T=init_exp_T,
        )

        if bars is not None and not bars.empty:
            bars["download_date"] = date_str
            if verbose:
                print(f"   ✅ {symbol} {date_str}: {len(bars)} bars")
            return bars

        if verbose:
            print(f"   ⚠️  {symbol} {date_str}: {0 if bars is None else len(bars)} bars (below threshold)")
        return None

    except Exception as e:
        if verbose:
            print(f"   ❌ {symbol} {date_str}: {e}")
        return None


def _seed_context_from_history(ctx, symbol: str, output_dir: Path, earliest: str) -> None:
    """Prime a FeatureContext from existing training parquets (strictly prior).

    Looks for the most recent parquet whose last bar precedes ``earliest`` and
    seeds both the bars tail and the accumulating winsorizer. No-op when no
    history exists (first run).
    """
    import pandas as pd  # noqa: F401  (local import kept for clarity)

    candidates = []
    if output_dir.exists():
        for year_dir in sorted(output_dir.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            for f in sorted(year_dir.glob(f"{symbol}_*.parquet")):
                candidates.append(f)
    if not candidates:
        return

    cutoff = pd.Timestamp(earliest)
    for path in reversed(candidates):
        try:
            prior = pd.read_parquet(path)
        except Exception:
            continue
        if prior.empty or "open_time" not in prior.columns:
            continue
        prior = prior[pd.to_datetime(prior["open_time"]) < cutoff]
        if prior.empty:
            continue
        ctx.seed_bars(prior)
        ctx.seed_winsorizer(prior)
        return


def main():
    parser = argparse.ArgumentParser(
        description="Process specific days through DRB pipeline with failed-day recovery"
    )
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument(
        "--dates", nargs="+", required=True,
        help="One or more dates in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--experiments-dir", default="experiments/",
        help="Path to experiments directory with Optuna studies"
    )
    parser.add_argument(
        "--output-dir", default="data_optimized/training/",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--source", type=str, default="binance", choices=["binance", "local", "auto"],
        help="Data source: binance (download), local (parquet), auto (local + fallback)"
    )
    parser.add_argument(
        "--data-dir", default="data_raw",
        help="Directory where raw tick parquet files are stored"
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Number of trials for failed days optimization (default: 30)"
    )
    parser.add_argument(
        "--no-positioning", action="store_true",
        help="Skip positioning features (funding rate, OI, taker ratio, L/S ratio)"
    )
    parser.add_argument(
        "--positioning-data-dir", default="data_raw/futures",
        help="Directory for caching Binance Futures data"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without processing"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = DEFAULT_SYMBOLS

    dates = args.dates
    output_dir = Path(args.output_dir)
    experiments_dir = Path(args.experiments_dir)
    verbose = not args.quiet
    positioning = not args.no_positioning

    for d in dates:
        try:
            parse_date(d)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    print("\n" + "⚙️" * 20)
    print("  PROCESS SPECIFIC DAYS — DRB Pipeline")
    print(f"  Symbols : {', '.join(symbols)}")
    print(f"  Dates   : {len(dates)} day(s)")
    for d in dates:
        print(f"           • {d}")
    print(f"  Source  : {args.source}")
    print(f"  Trials  : {args.trials} (for failed-day recovery)")
    print(f"  Positioning: {'enabled' if positioning else 'disabled'}")
    print(f"  Output  : {output_dir}")
    print("⚙️" * 20 + "\n")

    loader = HyperparamLoader(experiments_dir)
    positioner = PositioningCalculator(data_dir=args.positioning_data_dir) if positioning else None

    for symbol in symbols:
        if args.dry_run:
            print(f"\n🔍 DRY RUN: {symbol}")
            dates_by_month = defaultdict(list)
            for d in dates:
                y, m, day = parse_date(d)
                dates_by_month[(y, m)].append(d)

            for (y, m), month_dates in sorted(dates_by_month.items()):
                hypermeta = loader.get_params_with_meta(symbol, y, m, verbose=True)
                print(f"   {y}-{m:02d}: {len(month_dates)} day(s), "
                      f"study={hypermeta.study_name}, "
                      f"λ={hypermeta.exp_lambda:.4f}, T={hypermeta.init_exp_T}")
            continue

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  📈 Processing {symbol}")
            print("=" * 60)

        # Group dates by (year, month) for per-month processing
        dates_by_month = defaultdict(list)
        for d in dates:
            y, m, _ = parse_date(d)
            dates_by_month[(y, m)].append(d)

        all_good_bars = []
        all_failed_dates = set()
        failed_by_month: dict[tuple[int, int], list[str]] = {}

        for (y, m), month_dates in sorted(dates_by_month.items()):
            if verbose:
                print(f"\n  📅 {y}-{m:02d}: {len(month_dates)} day(s)")

            hypermeta = loader.get_params_with_meta(symbol, y, m, verbose=verbose)

            params = {
                "exp_lambda": hypermeta.exp_lambda,
                "init_exp_T": hypermeta.init_exp_T,
            }

            month_failed = []
            for date_str in month_dates:
                bars = process_single_day(
                    symbol=symbol,
                    date_str=date_str,
                    exp_lambda=params["exp_lambda"],
                    init_exp_T=params["init_exp_T"],
                    source=args.source,
                    data_dir=args.data_dir,
                    verbose=verbose,
                )

                if bars is not None and len(bars) >= MIN_BARS_THRESHOLD:
                    all_good_bars.append(bars)
                else:
                    month_failed.append(date_str)
                    all_failed_dates.add(date_str)

            if month_failed:
                failed_by_month[(y, m)] = month_failed
                if verbose:
                    print(f"   ⚠️  {len(month_failed)} day(s) failed in {y}-{m:02d}: {month_failed}")

        # Recover failed days per month (bars only; features are computed
        # causally afterwards through a shared FeatureContext).
        recovered_by_month = {}
        recovered_dates_by_month = {}

        if failed_by_month:
            if verbose:
                print("\n  🔧 Recovering failed days...")

            for (y, m), month_failed in sorted(failed_by_month.items()):
                recovery_result = reprocess_failed_days(
                    symbol=symbol,
                    failed_dates=month_failed,
                    year=y,
                    month=m,
                    experiments_dir=experiments_dir,
                    verbose=verbose,
                    source=args.source,
                    data_dir=args.data_dir,
                    n_trials=args.trials,
                )

                recovered_dates = set(recovery_result["recovered_dates"])
                still_failed = set(month_failed) - recovered_dates
                all_failed_dates = (all_failed_dates - recovered_dates) | still_failed

                if not recovery_result["recovered_bars"].empty:
                    recovered_by_month[(y, m)] = recovery_result["recovered_bars"]
                    recovered_dates_by_month[(y, m)] = recovered_dates
                    if verbose:
                        print(f"   📊 Recovered {len(recovery_result['recovered_bars']):,} bars from {len(recovered_dates)} days")

        # Group good bars by month
        good_by_month = defaultdict(list)
        for bars in all_good_bars:
            y, m, _ = parse_date(bars["download_date"].iloc[0] if "download_date" in bars.columns else dates[0])
            good_by_month[(y, m)].append(bars)

        if not good_by_month and not recovered_by_month:
            if verbose:
                print(f"\n⚠️  [{symbol}] No data generated")
            continue

        # Causal feature pass: months in chronological order, rolling windows
        # and winsorizer bounds only ever contain strictly-prior data — no
        # global-quantile leakage (replaces the legacy winsorize=True paths).
        from scripts.build_yearly_training_data import get_failed_days_optimized_params
        from src.pipeline.feature_context import FeatureContext

        ctx = FeatureContext(symbol=symbol)

        # Seed the causal context with existing training history strictly before
        # the earliest requested date, so isolated days get real rolling-window
        # history and past-only winsorizer bounds instead of warm-up dropout.
        _seed_context_from_history(ctx, symbol, output_dir, min(dates))

        all_feature_frames = []
        months_union = sorted(set(good_by_month) | set(recovered_by_month))

        for (y, m) in months_union:
            month_bars_list = list(good_by_month.get((y, m), []))
            recovered_bars = recovered_by_month.get((y, m))
            if recovered_bars is not None:
                month_bars_list.append(recovered_bars)
            if not month_bars_list:
                continue

            month_bars = pd.concat(month_bars_list, ignore_index=True)
            if "download_date" in month_bars.columns:
                month_bars = month_bars.drop(columns=["download_date"])
            month_bars = month_bars.sort_values("open_time").reset_index(drop=True)

            feat_kwargs = {"positioning_data_dir": args.positioning_data_dir}
            if positioner is not None:
                try:
                    pos_data = positioner.download_positioning_data(symbol, y, m)
                    feat_kwargs["positioning_data"] = pos_data
                except Exception:
                    pass

            features = ctx.compute_features(month_bars, **feat_kwargs)
            if features is None or features.empty:
                continue

            features["symbol"] = symbol
            features["year"] = np.int16(y)
            features["month"] = np.int8(m)

            # Per-bar metadata: recovered dates get the failed-day study,
            # everything else gets the walk-forward study.
            rec_dates = recovered_dates_by_month.get((y, m), set())
            hypermeta = loader.get_params_with_meta(symbol, y, m, verbose=False)
            bar_dates = pd.to_datetime(features["open_time"]).dt.date.astype(str)
            is_recovered = bar_dates.isin(rec_dates).to_numpy()

            failed_params = get_failed_days_optimized_params(experiments_dir, symbol, y, m) if rec_dates else None
            f_lambda = failed_params["exp_lambda"] if failed_params else hypermeta.exp_lambda
            f_T = failed_params["init_exp_T"] if failed_params else hypermeta.init_exp_T

            features["exp_lambda"] = np.float32(np.where(is_recovered, f_lambda, hypermeta.exp_lambda))
            features["init_exp_T"] = np.int32(np.where(is_recovered, f_T, hypermeta.init_exp_T))
            features["study_source"] = np.where(is_recovered, f"failed_days_{y}_{m:02d}", hypermeta.study_name)
            features["completion_rate"] = np.float32(np.where(is_recovered, 1.0, hypermeta.completion_rate))

            all_feature_frames.append(features)
            if verbose:
                print(f"   🧮 {symbol} {y}-{m:02d}: {len(features):,} features (causal context)")

        if not all_feature_frames:
            if verbose:
                print(f"\n⚠️  [{symbol}] No features computed")
            continue

        full_df = pd.concat(all_feature_frames, ignore_index=True)
        full_df = full_df.sort_values("open_time").reset_index(drop=True)

        # Mark failed days
        if all_failed_dates:
            full_df = mark_failed_days(full_df, all_failed_dates)
        else:
            full_df["failed_day"] = np.int8(0)
            full_df["sample_weight"] = np.float32(SAMPLE_WEIGHT_NORMAL)

        full_df["symbol"] = full_df["symbol"].astype("category")
        full_df["study_source"] = full_df["study_source"].astype("category")

        # Determine output filename from date range
        date_range = sorted(dates)
        start_label = date_range[0].replace("-", "")
        end_label = date_range[-1].replace("-", "")
        date_label = f"{start_label}_to_{end_label}" if len(dates) > 1 else start_label

        out_path = output_dir / f"{symbol}_{date_label}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

        total_bars = len(full_df)
        failed_bars = int(full_df["failed_day"].sum())
        file_size_mb = out_path.stat().st_size / (1024 ** 2)
        nan_count = full_df.isna().sum().sum()
        nan_pct = nan_count / full_df.size * 100

        if verbose:
            print(f"\n   💾 Saved → {out_path}")
            print(f"      Bars      : {total_bars:,}")
            print(f"      Failed    : {failed_bars:,} ({failed_bars/total_bars*100:.1f}%)" if total_bars else "")
            print(f"      NaN %     : {nan_pct:.2f}%")
            print(f"      Size      : {file_size_mb:.1f} MB")

        del full_df, all_feature_frames
        gc.collect()

    print("\n✅ Process complete!\n")


if __name__ == "__main__":
    main()
