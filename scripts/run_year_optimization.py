#!/usr/bin/env python3
"""
Year-Based Bayesian Optimization Runner
Orchestrates 6 × 2-month window optimizations for a full calendar year.

For year Y, the schedule is:
  W0 (bootstrap): target_month=12 of Y-1  (data: Nov-Dec Y-1) — only if no prev year W6
  W1: target_month=01  (data: Dec Y-1 + Jan Y)
  W2: target_month=03  (data: Feb-Mar Y)
  W3: target_month=05  (data: Apr-May Y)
  W4: target_month=07  (data: Jun-Jul Y)
  W5: target_month=09  (data: Aug-Sep Y)
  W6: target_month=11  (data: Oct-Nov Y)

Each window loads ONLY its 2-month data slice into RAM (not the whole year).
Each window automatically uses the previous window's results as --prev-study-dir
for adaptive search bounds (D2).

For subsequent years (e.g. 2024), W6 of the previous year (bayesian_2023_11_w2m)
automatically feeds into W1 — no extra bootstrap needed.

Usage:
    python scripts/run_year_optimization.py --year 2023
    python scripts/run_year_optimization.py --year 2023 --start-window 4 --trials 150
    python scripts/run_year_optimization.py --year 2023 --dry-run
    python scripts/run_year_optimization.py --year 2023 --no-bootstrap
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, date, timedelta


# 6 windows per year (2-month steps, each loads only 2 months into RAM)
WINDOWS = [
    {"target_month": 1,  "label": "W1 (Jan)", "data_desc": "Dec prev + Jan"},
    {"target_month": 3,  "label": "W2 (Mar)", "data_desc": "Feb-Mar"},
    {"target_month": 5,  "label": "W3 (May)", "data_desc": "Apr-May"},
    {"target_month": 7,  "label": "W4 (Jul)", "data_desc": "Jun-Jul"},
    {"target_month": 9,  "label": "W5 (Sep)", "data_desc": "Aug-Sep"},
    {"target_month": 11, "label": "W6 (Nov)", "data_desc": "Oct-Nov"},
]

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]


def get_experiment_dir(year: int, month: int, window: int = 2) -> Path:
    """Returns the experiment directory path for a given year/month/window."""
    return Path(f"experiments/bayesian_{year}_{month:02d}_w{window}m")


def is_window_complete(exp_dir: Path, symbols: list = None) -> bool:
    """Check if a window has completed studies for all symbols."""
    if symbols is None:
        symbols = SYMBOLS
    if not exp_dir.exists():
        return False
    for symbol in symbols:
        pkl = exp_dir / symbol / f"{symbol}_optuna_study.pkl"
        if not pkl.exists():
            return False
    return True


def get_bootstrap_dir(year: int) -> Path:
    """Returns the bootstrap study directory (Dec of prev year)."""
    return get_experiment_dir(year - 1, 12)


def needs_bootstrap(year: int, symbols: list = None) -> bool:
    """
    W1 needs a previous study for adaptive bounds.
    If the previous year's W6 (month=11) exists, W1 chains from there.
    Otherwise, we need a bootstrap study on Nov-Dec of prev year (month=12).
    """
    if symbols is None:
        symbols = SYMBOLS
    prev_w6 = get_experiment_dir(year - 1, 11)
    bootstrap = get_bootstrap_dir(year)
    return not is_window_complete(prev_w6, symbols) and not is_window_complete(bootstrap, symbols)


def get_prev_study_dir(year: int, window_idx: int) -> Path | None:
    """
    Returns the previous window's experiment dir for adaptive bounds chaining.
    For W1, chains from: prev year W6 → bootstrap (Dec prev year) → None.
    """
    if window_idx == 0:
        # W1: try prev year's W6 first, then bootstrap
        prev_w6 = get_experiment_dir(year - 1, 11)
        if prev_w6.exists():
            return prev_w6
        bootstrap = get_bootstrap_dir(year)
        return bootstrap if bootstrap.exists() else None
    else:
        prev_month = WINDOWS[window_idx - 1]["target_month"]
        prev_dir = get_experiment_dir(year, prev_month)
        return prev_dir if prev_dir.exists() else None


def _run_optimization_command(
    year: int,
    month: int,
    label: str,
    n_trials: int,
    parallel: int,
    prev_dir: Path | None,
    dry_run: bool,
    source: str = "auto",
    data_dir: str = "data_raw",
    symbols: str = "BTCUSDT",
    symbols_list: list = None,
    allow_partial: bool = False,
    cutoff_date: str | None = None,
    sampler: str | None = None,
    device: str | None = None,
) -> bool:
    """Execute the tuning script for a single window via subprocess."""
    exp_dir = get_experiment_dir(year, month)

    # Parse symbols list for completion check (symbols is a comma-separated string for subprocess)
    check_symbols = symbols_list or [s.strip() for s in symbols.split(',') if s.strip()]

    # Check if already complete (skip unless partial is allowed)
    if not allow_partial and is_window_complete(exp_dir, check_symbols):
        print(f"  ⏭️  Already complete — skipping")
        return True

    if dry_run:
        print(f"  🔍 DRY RUN — would execute optimization here")
        return True

    cmd = [
        sys.executable,
        "optimization/tune_multiasset_hyperparams.py",
        "--year", str(year),
        "--month", str(month),
        "--window", "2",
        "--trials", str(n_trials),
        "--parallel", str(parallel),
        "--source", source,
        "--data-dir", data_dir,
        "--symbols", symbols,
    ]

    if sampler:
        cmd.extend(["--sampler", sampler])
    if device:
        cmd.extend(["--device", device])

    if allow_partial and cutoff_date:
        cmd.extend(["--cutoff-date", cutoff_date])

    if prev_dir:
        cmd.extend(["--prev-study-dir", str(prev_dir)])

    print(f"\n  🚀 Command: {' '.join(cmd)}\n")

    start_time = time.time()

    try:
        subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent.parent),
            check=True,
        )
        elapsed = time.time() - start_time
        print(f"\n  ✅ {label} completed in {elapsed/60:.1f} min")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n  ❌ {label} FAILED after {elapsed/60:.1f} min (exit code {e.returncode})")
        return False

    except KeyboardInterrupt:
        print(f"\n  ⚠️  Interrupted by user")
        return False


def run_bootstrap(year: int, n_trials: int, parallel: int, dry_run: bool, source: str, data_dir: str, symbols: str, symbols_list: list = None, sampler: str | None = None, device: str | None = None) -> bool:
    """
    Run bootstrap study: Nov-Dec of previous year.
    This gives W1 adaptive bounds when no previous year's W6 exists.
    """
    bootstrap_year = year - 1
    bootstrap_month = 12  # target=Dec, window=2 → loads Nov-Dec
    bootstrap_dir = get_experiment_dir(bootstrap_year, bootstrap_month)

    print(f"\n{'═' * 80}")
    print(f"  W0 (Bootstrap) — Nov-Dec {bootstrap_year}")
    print(f"  Target Month: {bootstrap_year}-{bootstrap_month:02d}")
    print(f"  Data Coverage: Nov-Dec {bootstrap_year}")
    print(f"  Output Dir: {bootstrap_dir}")
    print(f"  Prev Study: None (default bounds — first study)")
    print(f"{'═' * 80}")

    return _run_optimization_command(
        year=bootstrap_year,
        month=bootstrap_month,
        label=f"W0 (Bootstrap Nov-Dec {bootstrap_year})",
        n_trials=n_trials,
        parallel=parallel,
        prev_dir=None,
        dry_run=dry_run,
        source=source,
        data_dir=data_dir,
        symbols=symbols,
        symbols_list=symbols_list,
        allow_partial=False,
        cutoff_date=None,
        sampler=sampler,
        device=device,
    )


def run_window(
    year: int,
    window_idx: int,
    n_trials: int,
    parallel: int,
    dry_run: bool = False,
    source: str = "auto",
    data_dir: str = "data_raw",
    symbols: str = "BTCUSDT",
    symbols_list: list = None,
    allow_partial: bool = False,
    cutoff_date: str | None = None,
    sampler: str | None = None,
    device: str | None = None,
) -> bool:
    """Run a single optimization window."""
    w = WINDOWS[window_idx]
    month = w["target_month"]
    exp_dir = get_experiment_dir(year, month)
    prev_dir = get_prev_study_dir(year, window_idx)

    print(f"\n{'═' * 80}")
    print(f"  {w['label']} — Year {year}")
    print(f"  Target Month: {year}-{month:02d}")
    print(f"  Data Coverage: {w['data_desc']}")
    print(f"  Output Dir: {exp_dir}")
    print(f"  Prev Study: {prev_dir or 'None (default bounds)'}")
    if allow_partial and cutoff_date:
        print(f"  Cutoff Date: {cutoff_date}")
    print(f"{'═' * 80}")

    return _run_optimization_command(
        year=year,
        month=month,
        label=w["label"],
        n_trials=n_trials,
        parallel=parallel,
        prev_dir=prev_dir,
        dry_run=dry_run,
        source=source,
        data_dir=data_dir,
        symbols=symbols,
        symbols_list=symbols_list,
        allow_partial=allow_partial,
        cutoff_date=cutoff_date,
        sampler=sampler,
        device=device,
    )


def print_year_status(year: int, symbols: list = None):
    """Print status of all windows for a year including bootstrap."""
    if symbols is None:
        symbols = SYMBOLS

    print(f"\n{'━' * 80}")
    print(f"  📅 Year {year} — Window Status")
    print(f"{'━' * 80}")

    # Bootstrap status
    bootstrap_dir = get_bootstrap_dir(year)
    prev_w6 = get_experiment_dir(year - 1, 11)
    has_bootstrap = is_window_complete(bootstrap_dir, symbols)
    has_prev_w6 = is_window_complete(prev_w6, symbols)
    bootstrap_needed = not has_prev_w6 and not has_bootstrap

    if has_prev_w6:
        print(f"  W0 Bootstrap | ✅ Not needed  | {year-1} W6 exists ({prev_w6.name})")
    elif has_bootstrap:
        print(f"  W0 Bootstrap | ✅ Complete     | Nov-Dec {year-1} ({bootstrap_dir.name})")
    else:
        print(f"  W0 Bootstrap | ⏳ Needed      | Nov-Dec {year-1} (will run automatically)")

    complete = 0
    for i, w in enumerate(WINDOWS):
        exp_dir = get_experiment_dir(year, w["target_month"])
        done = is_window_complete(exp_dir, symbols)
        if done:
            complete += 1
        status = "✅ Complete" if done else "⏳ Pending"
        prev = get_prev_study_dir(year, i)
        chain = f"← {prev.name}" if prev else "← (default bounds)"
        print(f"  {w['label']:12s} | {status:14s} | {w['data_desc']:25s} | {chain}")

    print(f"{'━' * 80}")
    print(f"  Progress: {complete}/6 windows complete")
    print(f"{'━' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Year-Based Bayesian Optimization Runner — "
                    "Runs 6 × 2-month windows to cover a full calendar year."
    )
    parser.add_argument(
        "--year", type=int, required=True,
        help="Year to optimize (e.g. 2023)"
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Optuna trials per symbol per window (default: 100)"
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Symbols to process in parallel within each window (default: 1)"
    )
    parser.add_argument(
        "--start-window", type=int, default=0, choices=range(0, 7),
        help="Window to start from, 0=bootstrap, 1-6=W1-W6 (default: 0). "
             "Useful for resuming."
    )
    parser.add_argument(
        "--no-bootstrap", action="store_true",
        help="Skip the W0 bootstrap even if needed (W1 will use default bounds)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the execution plan without running optimizations"
    )
    parser.add_argument(
        "--source", type=str, default="auto", choices=["binance", "local", "auto"],
        help="Data source: binance (download), local (parquet), or auto (local + fallback)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data_raw",
        help="Directory where raw tick parquet files are stored"
    )
    parser.add_argument(
        "--symbols", type=str, default="BTCUSDT",
        help="Comma-separated symbols to process (e.g., BTCUSDT or BTCUSDT,ETHUSDT)"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow partial windows using --cutoff-date (bypasses completion skip)"
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Cutoff date for partial window (YYYY-MM-DD). Default: today-1 when --allow-partial"
    )
    parser.add_argument(
        "--sampler", type=str, default=None, choices=["tpe", "botorch"],
        help="Forward --sampler to the tuning script (default: tuning script default)"
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"],
        help="Forward --device to the tuning script (default: tuning script default)"
    )

    args = parser.parse_args()

    # Parse symbols from comma-separated string
    SYMBOLS = [s.strip() for s in args.symbols.split(',') if s.strip()]

    year = args.year
    start_idx = max(0, args.start_window - 1)  # -1 because W1=index 0 in WINDOWS
    skip_bootstrap = args.no_bootstrap or args.start_window > 1

    print("🗓️ " * 30)
    print("  YEAR-BASED BAYESIAN OPTIMIZATION")
    print("🗓️ " * 30)
    print(f"  Year:           {year}")
    print(f"  Trials/symbol:  {args.trials}")
    print(f"  Parallel:       {args.parallel}")
    print(f"  Start Window:   {'W0 (bootstrap)' if args.start_window == 0 else f'W{args.start_window}'}")
    print(f"  Source:         {args.source}")
    print(f"  Mode:           {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Symbols:        {', '.join(SYMBOLS)}")
    if args.allow_partial:
        print(f"  Partial:        enabled")
    print(f"  Started:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Show current status
    print_year_status(year, symbols=SYMBOLS)

    # Run
    total_start = time.time()
    results = {}

    # Bootstrap (W0): run Nov-Dec of prev year if needed
    if not skip_bootstrap and needs_bootstrap(year, SYMBOLS):
        print("\n📌 W0 Bootstrap is needed — running Nov-Dec " + str(year - 1))
        success = run_bootstrap(year, args.trials, args.parallel, args.dry_run, args.source, args.data_dir, args.symbols, symbols_list=SYMBOLS, sampler=args.sampler, device=args.device)
        results["W0 (Bootstrap)"] = success
        if not success and not args.dry_run:
            print(f"\n⚠️  Bootstrap failed. W1 will use default bounds.")
            # Don't stop — W1 can still run with default bounds
    elif not skip_bootstrap:
        print(f"\n✅ Bootstrap not needed — previous study exists for W1")

    # Run W1-W6 sequentially
    for i in range(start_idx, len(WINDOWS)):
        cutoff_date = args.cutoff_date
        if args.allow_partial and not cutoff_date:
            cutoff_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        success = run_window(
            year=year,
            window_idx=i,
            n_trials=args.trials,
            parallel=args.parallel,
            dry_run=args.dry_run,
            source=args.source,
            data_dir=args.data_dir,
            symbols=args.symbols,
            symbols_list=SYMBOLS,
            allow_partial=args.allow_partial,
            cutoff_date=cutoff_date,
            sampler=args.sampler,
            device=args.device,
        )
        results[WINDOWS[i]["label"]] = success

        if not success and not args.dry_run:
            print(f"\n⚠️  Stopping after failure in {WINDOWS[i]['label']}")
            print("  Use --start-window to resume from this window.")
            break

    # Final summary
    total_elapsed = time.time() - total_start

    print(f"\n{'🎉' * 30}")
    print("  YEAR OPTIMIZATION SUMMARY")
    print(f"{'🎉' * 30}")

    for label, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {label}")

    print(f"\n  Total time: {total_elapsed/60:.1f} min")
    print(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Show updated status
    print_year_status(year, symbols=SYMBOLS)


if __name__ == "__main__":
    main()
