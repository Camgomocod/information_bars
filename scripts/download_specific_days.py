#!/usr/bin/env python3
"""
download_specific_days.py — Descarga días específicos y los guarda en parquet mensual

Para automatización: descarga fechas exactas, agrupa por mes, y mergea con parquets
existentes sin duplicar días.

Usage:
    # Días específicos
    micromamba run -n trading-core python scripts/download_specific_days.py \
        --symbol BTCUSDT --dates 2026-01-15 2026-01-20

    # Múltiples símbolos
    micromamba run -n trading-core python scripts/download_specific_days.py \
        --symbols BTCUSDT ETHUSDT --dates 2026-01-15

    # Solo verificar (dry-run)
    micromamba run -n trading-core python scripts/download_specific_days.py \
        --symbol BTCUSDT --dates 2026-01-15 --dry-run

    # Output personalizado
    micromamba run -n trading-core python scripts/download_specific_days.py \
        --symbol BTCUSDT --dates 2026-01-15 --output-dir /mnt/data/raw
"""

import sys
import argparse
from pathlib import Path
import warnings
from collections import defaultdict

import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.connectors.download_data import DownloadData


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
DEFAULT_OUTPUT_DIR = "data_raw"


def parse_date(date_str: str) -> tuple[int, int, int]:
    parts = date_str.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")
    return int(parts[0]), int(parts[1]), int(parts[2])


def download_single_day(
    symbol: str,
    date_str: str,
    verbose: bool = True,
) -> pd.DataFrame | None:
    downloader = DownloadData(symbol=symbol)
    try:
        df_raw = downloader.download_day(date_str)
        if df_raw is None or df_raw.empty:
            if verbose:
                print(f"   ❌ {symbol} {date_str}: No data")
            return None
        df_raw["download_date"] = date_str
        if verbose:
            print(f"   ✅ {symbol} {date_str}: {len(df_raw):,} ticks")
        return df_raw
    except Exception as e:
        if verbose:
            print(f"   ❌ {symbol} {date_str}: {str(e)[:80]}")
        return None


def merge_into_monthly_parquet(
    month_df: pd.DataFrame,
    year: int,
    month: int,
    symbol: str,
    output_dir: Path,
    verbose: bool = True,
) -> Path:
    year_dir = output_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    out_path = year_dir / f"{symbol}_{year}_{month:02d}.parquet"

    if out_path.exists():
        # Only read download_date column to check for existing days (memory-efficient)
        existing_dates = pd.read_parquet(out_path, columns=["download_date"])
        existing_dates = set(existing_dates["download_date"].unique())
        new_dates = set(month_df["download_date"].unique())
        only_new = new_dates - existing_dates
        if not only_new:
            if verbose:
                print(f"   ⏭️  {out_path.name}: days already exist, skipping")
            return out_path

        # Use PyArrow to append new rows without loading full dataset into pandas
        import pyarrow.parquet as pq
        import pyarrow as pa
        existing_table = pq.read_table(out_path)
        new_table = pa.Table.from_pandas(month_df[month_df["download_date"].isin(only_new)])
        combined = pa.concat_tables([existing_table, new_table])
        pq.write_table(combined, out_path, compression="snappy")
        total_ticks = combined.num_rows
        del existing_table, new_table, combined
    else:
        month_df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        total_ticks = len(month_df)

    file_size_mb = out_path.stat().st_size / (1024 ** 2)
    if verbose:
        print(f"   💾 {out_path.name} ({total_ticks:,} ticks, {file_size_mb:.1f} MB)")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Download specific days of raw tick data to monthly parquet files"
    )
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument(
        "--dates", nargs="+", required=True,
        help="One or more dates in YYYY-MM-DD format"
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-day output")

    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = DEFAULT_SYMBOLS

    dates = args.dates
    output_dir = Path(args.output_dir)
    verbose = not args.quiet

    validated_dates = []
    for d in dates:
        try:
            parse_date(d)
            validated_dates.append(d)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    print("\n" + "📥" * 20)
    print(f"  SPECIFIC DAYS DOWNLOAD")
    print(f"  Symbols : {', '.join(symbols)}")
    print(f"  Dates   : {len(validated_dates)} day(s)")
    for d in validated_dates:
        print(f"           • {d}")
    print(f"  Output  : {output_dir}")
    print("📥" * 20 + "\n")

    if args.dry_run:
        print("🔍 Dry run - would download:")
        for symbol in symbols:
            for d in validated_dates:
                y, m, day = parse_date(d)
                out_path = output_dir / str(y) / f"{symbol}_{y}_{m:02d}.parquet"
                exists = out_path.exists()
                print(f"   {symbol} {d} → {out_path.name} {'(exists)' if exists else '(new)'}")
        return

    by_month: dict[tuple[str, int, int], list[pd.DataFrame]] = defaultdict(list)
    total_downloaded = 0
    total_failed = 0

    for symbol in symbols:
        for date_str in validated_dates:
            df = download_single_day(symbol, date_str, verbose=verbose)
            if df is not None:
                y, m, _ = parse_date(date_str)
                by_month[(symbol, y, m)].append(df)
                total_downloaded += 1
            else:
                total_failed += 1

    if not by_month:
        print("\n⚠️  No data downloaded.")
        sys.exit(1)

    print()
    for (symbol, y, m), dfs in sorted(by_month.items()):
        month_df = pd.concat(dfs, ignore_index=True)
        merge_into_monthly_parquet(month_df, y, m, symbol, output_dir, verbose=True)

    print("\n" + "=" * 60)
    print(f"  ✅ Complete: {total_downloaded} days OK, {total_failed} failed")
    print("=" * 60 + "\n")

    if total_failed > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
