"""
build_training_data.py — Walk-Forward Pipeline: Download → DRBs → Features → Parquet

Usage:
    micromamba run -n trading-core python scripts/build_training_data.py [OPTIONS]

Options:
    --symbols   Space-separated list (default: BTCUSDT ETHUSDT BNBUSDT SOLUSDT)
    --start     Start month YYYY-MM (default: 2023-01)
    --end       End month YYYY-MM (default: current month)
    --output-dir Output directory (default: data_optimized/training/)
    --experiments-dir  Experiments directory (default: experiments/)
    --dry-run   Only print the walk-forward plan, do not download data

Example — full run:
    micromamba run -n trading-core python scripts/build_training_data.py

Example — test on 2 months of BTCUSDT:
    micromamba run -n trading-core python scripts/build_training_data.py \
        --symbols BTCUSDT --start 2023-01 --end 2023-02
"""

import sys
import argparse
from pathlib import Path
from datetime import date
from calendar import monthrange
import warnings
import gc

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline.hyperparam_loader import HyperparamLoader
from src.pipeline.bar_builder import build_monthly_bars
from src.features.base_features import compute_all_features
from src.features.positioning_calculator import PositioningCalculator


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def month_range(start_ym: str, end_ym: str) -> list[tuple[int, int]]:
    """Returns list of (year, month) tuples from start to end inclusive."""
    sy, sm = int(start_ym[:4]), int(start_ym[5:7])
    ey, em = int(end_ym[:4]), int(end_ym[5:7])
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return months


def print_summary_table(summary: list[dict]):
    """Print a nice summary table of results."""
    print("\n" + "=" * 90)
    print("📊 PIPELINE SUMMARY")
    print("=" * 90)
    header = f"{'Symbol':<10} {'Period':<20} {'Bars':>8} {'Features':>10} {'NaN%':>7} {'File Size':>12}"
    print(header)
    print("-" * 90)
    for row in summary:
        print(
            f"{row['symbol']:<10} "
            f"{row['period']:<20} "
            f"{row['total_bars']:>8,} "
            f"{row['feature_cols']:>10} "
            f"{row['nan_pct']:>6.1f}% "
            f"{row['file_size_mb']:>10.1f} MB"
        )
    print("=" * 90)


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    symbols: list[str],
    start_ym: str,
    end_ym: str,
    output_dir: Path,
    experiments_dir: Path,
    dry_run: bool = False,
    positioning: bool = True,
    positioning_data_dir: str = "data_raw/futures",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    months = month_range(start_ym, end_ym)
    loader = HyperparamLoader(experiments_dir)

    summary_rows = []

    print("\n" + "🚀" * 20)
    print(f"  WALK-FORWARD TRAINING DATA PIPELINE")
    print(f"  Symbols : {', '.join(symbols)}")
    print(f"  Period  : {start_ym} → {end_ym}  ({len(months)} months)")
    print(f"  Output  : {output_dir}")
    print("🚀" * 20 + "\n")

    positioner = PositioningCalculator(data_dir=positioning_data_dir) if positioning else None

    for symbol in symbols:
        print(f"\n{'=' * 70}")
        print(f"  📈 Processing {symbol} ({len(months)} months)")
        print("=" * 70)

        if dry_run:
            # Just print the walk-forward plan
            print(f"\n{'Month':<15} {'Study Used':<35} {'Lambda':>10} {'T':>8}")
            print("-" * 70)
            for year, month in months:
                params = loader.get_params(symbol, year, month, verbose=False)
                study_folders = sorted(
                    [f for f in experiments_dir.iterdir() if f.is_dir()],
                    key=lambda p: p.name
                )
                # Find which study is used (most recent prior)
                target_date = date(year, month, 1)
                import re
                pat = re.compile(r"^bayesian_(\d{4})_(\d{2})_w2m$")
                best_study = "default"
                for f in sorted(study_folders, key=lambda p: p.name):
                    mm = pat.match(f.name)
                    if mm:
                        y2, m2 = int(mm.group(1)), int(mm.group(2))
                        if date(y2, m2, 1) < target_date:
                            best_study = f.name
                print(
                    f"  {year}-{month:02d}         "
                    f"{best_study:<35} "
                    f"{params['exp_lambda']:>10.4f} "
                    f"{params['init_exp_T']:>8}"
                )
            continue

        # ── Real run: build bars + features per month, then concat ──
        monthly_frames: list[pd.DataFrame] = []

        # Keep rolling feature history and past-only winsorizer bounds across
        # month boundaries. DRBs remain stateless; features are chronological.
        from src.pipeline.feature_context import FeatureContext
        feature_context = FeatureContext(symbol=symbol)

        for year, month in months:
            # 1. Get walk-forward hyperparams for this month
            params = loader.get_params(symbol, year, month)

            # 2. Download ticks → DRBs
            bars_df = build_monthly_bars(
                symbol=symbol,
                year=year,
                month=month,
                params=params,
                min_days=15,
            )

            if bars_df is None or bars_df.empty:
                print(f"   ⏭️  [{symbol}] {year}-{month:02d}: skipped (no bars)")
                continue

            # 3. Compute features (VectorBT-based, causal across months)
            kwargs = dict(positioning_data_dir=positioning_data_dir)
            if positioning and positioner is not None:
                try:
                    pos_data = positioner.download_positioning_data(symbol, year, month)
                    n_pos = sum(1 for v in pos_data.values() if v is not None and not v.empty)
                    print(f"   📊 [{symbol}] {year}-{month:02d}: Positioning data: {n_pos}/1 metrics loaded")
                    kwargs["positioning_data"] = pos_data
                    kwargs["symbol"] = symbol
                except Exception as e:
                    print(f"   ⚠️  [{symbol}] {year}-{month:02d}: Positioning data download failed: {e}")
                    print(f"   ⚠️  [{symbol}] {year}-{month:02d}: Continuing without positioning features")

            try:
                bars_features = feature_context.compute_features(bars_df, **kwargs)
            except Exception as e:
                print(f"   ❌ [{symbol}] {year}-{month:02d}: Feature error: {e}")
                continue

            if bars_features is None or bars_features.empty:
                continue

            # 4. Tag with symbol and period metadata
            bars_features["symbol"] = symbol
            bars_features["year"] = np.int16(year)
            bars_features["month"] = np.int8(month)

            monthly_frames.append(bars_features)
            print(f"   ✅ [{symbol}] {year}-{month:02d}: {len(bars_features):,} bars with features")

            # GC every month to keep memory in check
            del bars_df
            gc.collect()

        if not monthly_frames:
            print(f"\n⚠️  [{symbol}]: No data generated — skipping Parquet save")
            continue

        # ── Concatenate all months for this symbol ──
        print(f"\n   📦 [{symbol}] Concatenating {len(monthly_frames)} months...")
        full_df = pd.concat(monthly_frames, ignore_index=True)
        full_df["symbol"] = full_df["symbol"].astype("category")

        # ── Save Parquet (backup) ──
        out_path = output_dir / f"{symbol}_training.parquet"
        full_df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

        # ── Sync to TimescaleDB ──
        try:
            from src.storage.timescale_client import TimescaleDBClient
            db = TimescaleDBClient()
            db.write_bars(full_df, symbol=symbol)
            print(f"   📊 Synced {len(full_df):,} bars to TimescaleDB")
            db.close()
        except Exception as e:
            print(f"   ⚠️  DB sync failed (parquet preserved): {e}")

        file_size_mb = out_path.stat().st_size / (1024 ** 2)
        nan_count = full_df.isna().sum().sum()
        nan_pct = nan_count / full_df.size * 100
        total_cols = len([c for c in full_df.columns if c not in ("symbol", "year", "month")])

        period_str = f"{monthly_frames[0]['open_time'].iloc[0].strftime('%Y-%m')} → " \
                     f"{monthly_frames[-1]['close_time'].iloc[-1].strftime('%Y-%m')}" \
            if "open_time" in full_df.columns and "close_time" in full_df.columns else "N/A"

        summary_rows.append({
            "symbol": symbol,
            "period": period_str,
            "total_bars": len(full_df),
            "feature_cols": total_cols,
            "nan_pct": nan_pct,
            "file_size_mb": file_size_mb,
        })

        print(f"\n   💾 [{symbol}] Saved → {out_path}")
        print(f"        Bars    : {len(full_df):,}")
        print(f"        Columns : {len(full_df.columns)}")
        print(f"        NaN %   : {nan_pct:.2f}%")
        print(f"        Size    : {file_size_mb:.1f} MB")

        del full_df, monthly_frames
        gc.collect()

    if not dry_run:
        print_summary_table(summary_rows)
        print("\n✅ Pipeline complete!\n")


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def parse_args():
    today = date.today()
    default_end = f"{today.year}-{today.month:02d}"

    parser = argparse.ArgumentParser(
        description="Walk-Forward Training Data Pipeline: Download → DRBs → Features → Parquet"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
        help="Symbols to process (default: all 4)",
    )
    parser.add_argument(
        "--start",
        default="2023-01",
        help="Start month YYYY-MM (default: 2023-01)",
    )
    parser.add_argument(
        "--end",
        default=default_end,
        help=f"End month YYYY-MM (default: {default_end})",
    )
    parser.add_argument(
        "--output-dir",
        default="data_optimized/training/",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--experiments-dir",
        default="experiments/",
        help="Path to experiments directory containing Optuna studies",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print walk-forward plan without downloading data",
    )
    parser.add_argument(
        "--no-positioning",
        action="store_true",
        help="Skip positioning features (funding rate, OI, taker ratio, L/S ratio)",
    )
    parser.add_argument(
        "--positioning-data-dir",
        default="data_raw/futures",
        help="Directory for caching Binance Futures data (default: data_raw/futures)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        symbols=args.symbols,
        start_ym=args.start,
        end_ym=args.end,
        output_dir=Path(args.output_dir),
        experiments_dir=Path(args.experiments_dir),
        dry_run=args.dry_run,
        positioning=not args.no_positioning,
        positioning_data_dir=args.positioning_data_dir,
    )
