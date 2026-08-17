#!/usr/bin/env python3
"""
process_yesterday.py — Daily Inference Pipeline (VPS)

Processes yesterday's tick data for PREDICTION layer. Writes to a separate
inference-only location (data_optimized/inference/), NEVER mixed with training data.

Fallback chain guarantees at least 1 bar with real price data per trading day.

Usage:
    # Daily cron (recommended):
    0 1 * * * cd /path/to/trading-core && \\
        micromamba run -n trading-core python scripts/process_yesterday.py \\
        --symbols BTCUSDT ETHUSDT SOLUSDT --source binance --new-month optimize

    # Dry run:
    python scripts/process_yesterday.py --symbols BTCUSDT --dry-run

    # Specific date:
    python scripts/process_yesterday.py --symbols BTCUSDT --date 2026-05-26
"""

import sys
import argparse
from pathlib import Path
from datetime import date, timedelta
from typing import List

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Daily Inference Pipeline — process yesterday's tick data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process yesterday (inference):
  %(prog)s --symbols BTCUSDT ETHUSDT

  # Process yesterday from Binance only:
  %(prog)s --symbols BTCUSDT --source binance

  # Specific date:
  %(prog)s --symbols BTCUSDT --date 2026-05-26

  # Auto-optimize at month boundaries:
  %(prog)s --symbols BTCUSDT ETHUSDT --new-month optimize

  # Dry run:
  %(prog)s --symbols BTCUSDT --dry-run
        """,
    )

    parser.add_argument(
        "--symbols", nargs="+", required=True,
        help="Trading pairs (e.g. BTCUSDT ETHUSDT SOLUSDT)",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Specific date (YYYY-MM-DD). Default: yesterday.",
    )
    parser.add_argument(
        "--source", type=str, default="auto",
        choices=["auto", "binance", "local"],
        help="Data source. Default: auto.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data_optimized/inference/",
        help="Output directory for inference parquets. "
             "Default: data_optimized/inference/",
    )
    parser.add_argument(
        "--experiments-dir", type=str,
        default=str(project_root / "experiments"),
        help="Experiments directory. Default: <trading-core>/experiments",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data_raw",
        help="Raw data directory. Default: data_raw",
    )
    parser.add_argument(
        "--db-url", type=str, default=None,
        help="TimescaleDB connection string. Default: TRADING_CORE_DATABASE_URL / env vars.",
    )
    parser.add_argument(
        "--new-month", type=str, default="use-latest",
        choices=["optimize", "use-latest", "fail"],
        help="Strategy when study missing at month boundary. "
             "optimize: run inline. "
             "use-latest: use most recent available. "
             "fail: exit with error. "
             "Default: use-latest.",
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Optuna trials when auto-optimizing. Default: 50.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process but do NOT save.",
    )
    parser.add_argument(
        "--bootstrap", action="store_true",
        help="Seed inference parquet from training data before processing. "
             "Only needed on first run.",
    )
    parser.add_argument(
        "--bootstrap-bars", type=int, default=500,
        help="Number of bars to copy from training. Default: 500.",
    )
    parser.add_argument(
        "--training-dir", type=str, default="data_optimized/training/",
        help="Training parquet directory for bootstrap. "
             "Default: data_optimized/training/",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output.",
    )

    args = parser.parse_args()

    if args.date:
        try:
            target_date = date.fromisoformat(args.date)
        except ValueError:
            print(f"❌ Invalid date: {args.date}. Use YYYY-MM-DD.")
            return 2
    else:
        target_date = date.today() - timedelta(days=1)

    target_str = target_date.strftime("%Y-%m-%d")

    print(f"\n{'═' * 70}")
    print(f"  🚀 Inference Pipeline — {target_str}")
    print(f"  Symbols: {', '.join(args.symbols)}")
    print(f"  Source: {args.source}")
    print(f"  Output: {args.output_dir}")
    print(f"  New month: {args.new_month}")
    if args.dry_run:
        print(f"  ⚠️  DRY RUN")
    print(f"{'═' * 70}")

    from src.pipeline.daily_processor import DailyProcessor

    processor = DailyProcessor(
        output_dir=args.output_dir,
        experiments_dir=args.experiments_dir,
        db_url=args.db_url,
        data_dir=args.data_dir,
        verbose=not args.quiet,
        dry_run=args.dry_run,
    )

    has_failures = False

    if args.bootstrap:
        for symbol in args.symbols:
            processor.bootstrap_from_training(
                symbol=symbol,
                training_dir=args.training_dir,
                context_bars=args.bootstrap_bars,
            )

    for symbol in args.symbols:
        r = processor.process_day(
            symbol=symbol, target_date=target_date,
            source=args.source, new_month_strategy=args.new_month,
            optimize_trials=args.trials,
        )
        status_icon = "✓" if r.status in ("processed", "success") else "✗"
        n_bars = getattr(r, "n_bars", "?") if r else "?"
        print(f"  [S1] {symbol:<10} {status_icon} bars={n_bars} status={r.status if r else 'none'}")
        if r.status == "failed":
            has_failures = True

    print(f"\n{'═' * 70}")
    if has_failures:
        print(f"  Pipeline completed with failures")
    else:
        print(f"  Pipeline completed successfully")
    print(f"{'═' * 70}\n")

    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
