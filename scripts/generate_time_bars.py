"""
generate_time_bars.py — Generate Time Bars (1m/5m/15m/1h) from raw ticks

Usage:
    micromamba run -n trading-core python scripts/generate_time_bars.py \
        --symbols BTCUSDT --year 2024 --month 01

    micromamba run -n trading-core python scripts/generate_time_bars.py \
        --symbols BTCUSDT ETHUSDT --year 2024 \
        --intervals 1 15 60

    micromamba run -n trading-core python scripts/generate_time_bars.py \
        --symbols BTCUSDT --year 2024 --month 01 --dry-run
"""

import sys
import argparse
import gc
import warnings
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.normalizers.data_normalizer import DataNormalizer
from src.config import get_bar_config

DEFAULT_SYMBOLS = ["BTCUSDT"]
DEFAULT_INTERVALS = get_bar_config("standard").get("time_intervals", [1, 5, 15, 60])


def resolve_months(year: int, month: int | None) -> list[tuple[int, int]]:
    if month is not None:
        return [(year, month)]
    today = date.today()
    months = []
    for m in range(1, 13):
        if year == today.year and m > today.month:
            break
        months.append((year, m))
    return months


def load_monthly_ticks(symbol: str, year: int, month: int, data_dir: Path) -> pd.DataFrame:
    file_path = data_dir / str(year) / f"{symbol}_{year}_{month:02d}.parquet"
    if not file_path.exists():
        print(f"   ⚠️  {file_path} not found, skipping")
        return pd.DataFrame()
    df = pd.read_parquet(file_path)
    if df.empty:
        return df
    cols = [c for c in ["timestamp", "price", "quantity", "dollar_value"] if c in df.columns]
    return df[cols].copy()


def resample_to_time_bars(ticks: pd.DataFrame, interval_min: int) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame()

    df = ticks.set_index("timestamp").sort_index()

    rule = f"{interval_min}min"
    resampled = df.resample(rule)

    bars = pd.DataFrame({
        "open": resampled["price"].first(),
        "high": resampled["price"].max(),
        "low": resampled["price"].min(),
        "close": resampled["price"].last(),
        "volume": resampled["quantity"].sum(),
        "dollar_value": resampled["dollar_value"].sum() if "dollar_value" in df.columns else resampled["quantity"].apply(lambda x: 0.0),
        "n_ticks": resampled["price"].count(),
    })

    bars = bars.dropna(subset=["open", "high", "low", "close"])
    if bars.empty:
        return pd.DataFrame()

    bars = bars.reset_index()

    bars["open_time"] = bars["timestamp"]
    bars["close_time"] = bars["timestamp"] + pd.Timedelta(minutes=interval_min)

    bars["open"] = bars["open"].astype(np.float32)
    bars["high"] = bars["high"].astype(np.float32)
    bars["low"] = bars["low"].astype(np.float32)
    bars["close"] = bars["close"].astype(np.float32)
    bars["volume"] = bars["volume"].astype(np.float32)
    bars["dollar_value"] = bars["dollar_value"].astype(np.float32)
    bars["n_ticks"] = bars["n_ticks"].astype(np.int32)

    columns = [
        "open_time", "close_time",
        "open", "high", "low", "close",
        "volume", "dollar_value", "n_ticks",
    ]
    return bars[columns]


def process_month(
    symbol: str,
    year: int,
    month: int,
    intervals: list[int],
    data_dir: Path,
    output_dir: Path,
    normalizer: DataNormalizer,
    dry_run: bool = False,
) -> dict:
    month_str = f"{year}-{month:02d}"
    print(f"\n📅 {symbol} — {month_str}")
    print("-" * 50)

    ticks = load_monthly_ticks(symbol, year, month, data_dir)
    if ticks.empty:
        return {"symbol": symbol, "month": month_str, "bars": {}, "status": "no_data"}

    print(f"   Loaded {len(ticks):,} ticks")

    ticks = normalizer.clean_raw_ticks(ticks)
    print(f"   After cleaning: {len(ticks):,} ticks")

    if ticks.empty:
        return {"symbol": symbol, "month": month_str, "bars": {}, "status": "empty_after_clean"}

    results = {}
    symbol_dir = output_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    for interval in intervals:
        bars = resample_to_time_bars(ticks, interval)
        if bars.empty:
            results[interval] = 0
            print(f"   ⏱️  {interval:2d}min: 0 bars")
            continue

        if dry_run:
            results[interval] = len(bars)
            print(f"   ⏱️  {interval:2d}min: {len(bars):,} bars (dry-run, not saved)")
            continue

        bars["date"] = bars["open_time"].dt.date
        day_groups = _split_by_date(bars)
        print(f"   ⏱️  {interval:2d}min: {len(bars):,} bars ({len(day_groups)} days)")
        for day_bars in day_groups:
            date_str = day_bars["date"].iloc[0].strftime("%Y-%m-%d")
            out_path = symbol_dir / f"{symbol}_{date_str}_time_{interval}min.parquet"
            day_bars.drop(columns=["date"]).to_parquet(out_path, index=False)

        results[interval] = len(bars)

    return {"symbol": symbol, "month": month_str, "bars": results, "status": "ok"}


def _split_by_date(df: pd.DataFrame) -> list[pd.DataFrame]:
    return [g for _, g in df.groupby(df["open_time"].dt.date)]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Time Bars (1m/5m/15m/1h) from raw tick data"
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--year", type=int, default=date.today().year)
    parser.add_argument("--month", type=int, default=None, choices=range(1, 13))
    parser.add_argument("--intervals", nargs="+", type=int, default=DEFAULT_INTERVALS)
    parser.add_argument("--source", choices=["local", "binance", "auto"], default="local")
    parser.add_argument("--data-dir", default="data_raw")
    parser.add_argument("--output-dir", default="data/bars/time")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    normalizer = DataNormalizer()

    if args.source not in ("local", "auto"):
        print("❌ --source binance not yet implemented; use --source local")
        sys.exit(1)

    symbols = args.symbols
    months = resolve_months(args.year, args.month)

    print("\n" + "=" * 60)
    print("🚀 TIME BAR GENERATION")
    print("=" * 60)
    print(f"Symbols  : {', '.join(symbols)}")
    month_display = f"{args.month:02d}" if args.month else "01→12"
    print(f"Period   : {args.year}-{month_display}")
    print(f"Intervals: {', '.join(f'{i}min' for i in args.intervals)}")
    print(f"Source   : {data_dir}")
    print(f"Output   : {output_dir}")
    if args.dry_run:
        print(f"🔷 DRY-RUN — no files will be written")
    print("=" * 60 + "\n")

    summary = []
    for symbol in symbols:
        for year, month in months:
            result = process_month(
                symbol=symbol,
                year=year,
                month=month,
                intervals=args.intervals,
                data_dir=data_dir,
                output_dir=output_dir,
                normalizer=normalizer,
                dry_run=args.dry_run,
            )
            summary.append(result)
            gc.collect()

    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    total_bars = 0
    for s in summary:
        bars_str = ", ".join(f"{k}min:{v:,}" for k, v in s["bars"].items())
        status = s["status"]
        print(f"  {s['symbol']:<10} {s['month']:<10}  {bars_str:<40}  {status}")
        total_bars += sum(s["bars"].values())
    print("=" * 60)
    print(f"  Total bars generated: {total_bars:,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
