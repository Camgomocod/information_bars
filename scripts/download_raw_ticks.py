#!/usr/bin/env python3
"""
download_raw_ticks.py — Descarga ticks crudos y los guarda en parquet por año

Usage:
    # Un símbolo, un año
    micromamba run -n trading-core python scripts/download_raw_ticks.py \
        --symbol BTCUSDT --year 2023

    # Múltiples años
    micromamba run -n trading-core python scripts/download_raw_ticks.py \
        --symbol BTCUSDT --start-year 2023 --end-year 2025

    # Un mes específico
    micromamba run -n trading-core python scripts/download_raw_ticks.py \
        --symbol BTCUSDT --year 2026 --month 01

    # Múltiples símbolos
    micromamba run -n trading-core python scripts/download_raw_ticks.py \
        --symbols BTCUSDT ETHUSDT --year 2023

    # Solo verificar qué días faltan (dry-run)
    micromamba run -n trading-core python scripts/download_raw_ticks.py \
        --symbol BTCUSDT --year 2023 --dry-run
"""

import sys
import argparse
from pathlib import Path
import warnings
import gc
from datetime import date, timedelta
from calendar import monthrange

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.connectors.download_data import DownloadData
from src.normalizers.data_normalizer import DataNormalizer


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
DEFAULT_OUTPUT_DIR = "data_raw"


def get_days_in_month(year: int, month: int) -> list[str]:
    """Retorna lista de fechas (YYYY-MM-DD) del mes."""
    num_days = monthrange(year, month)[1]
    return [f"{year}-{month:02d}-{day:02d}" for day in range(1, num_days + 1)]


def get_target_days(year: int, month: int, today: date | None = None) -> list[str]:
    """Retorna dias objetivo del mes, limitado hasta hoy-1 y sin futuros."""
    if today is None:
        today = date.today()

    last_day = monthrange(year, month)[1]
    month_start = date(year, month, 1)
    month_end = date(year, month, last_day)
    cap_date = min(month_end, today - timedelta(days=1))

    if cap_date < month_start:
        return []

    days = []
    current = month_start
    while current <= cap_date:
        days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return days


def get_existing_dates(parquet_path: Path) -> set[str]:
    """Lee solo download_date desde parquet mensual, devuelve conjunto de fechas."""
    if not parquet_path.exists():
        return set()

    try:
        existing = pd.read_parquet(parquet_path, columns=["download_date"])
        return set(existing["download_date"].unique())
    except Exception:
        return set()


def download_month_ticks(
    symbol: str,
    year: int,
    month: int,
    output_dir: Path,
    normalizer: DataNormalizer,
    verbose: bool = True,
) -> dict:
    """Descarga todos los días de un mes y guarda en parquet."""
    year_dir = output_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    month_output = year_dir / f"{symbol}_{year}_{month:02d}.parquet"

    days = get_target_days(year, month)
    downloader = DownloadData(symbol=symbol)

    if not days:
        if verbose:
            print(f"   ⏭️  {year}-{month:02d}: mes futuro o sin dias cerrados")
        return {"downloaded": 0, "failed": 0, "file": None}

    existing_dates = get_existing_dates(month_output)
    missing_days = [d for d in days if d not in existing_dates]

    if not missing_days:
        if verbose:
            print(f"   ⏭️  {month_output.name}: todo al dia ({len(days)} dias)")
        return {"downloaded": 0, "failed": 0, "file": month_output}

    all_days_data = []
    downloaded = 0
    failed = 0

    for day_str in missing_days:
        try:
            df_raw = downloader.download_day(day_str)

            if df_raw is None or df_raw.empty:
                if verbose:
                    print(f"   ❌ {day_str}: No data")
                failed += 1
                continue

            # Guardar raw ticks (sin limpieza para preservar datos originales)
            df_raw["download_date"] = day_str
            all_days_data.append(df_raw)
            downloaded += 1

            if verbose:
                print(f"   ✅ {day_str}: {len(df_raw):,} ticks")

        except Exception as e:
            if verbose:
                print(f"   ❌ {day_str}: {str(e)[:50]}")
            failed += 1

    if not all_days_data:
        if verbose:
            print(f"   ⚠️  No se descargaron datos para {year}-{month:02d}")
        return {"downloaded": 0, "failed": failed, "file": None}

    # Concatenar y guardar (solo dias faltantes)
    month_df = pd.concat(all_days_data, ignore_index=True)

    if month_output.exists():
        import pyarrow.parquet as pq
        import pyarrow as pa

        existing_table = pq.read_table(month_output)
        new_table = pa.Table.from_pandas(
            month_df[month_df["download_date"].isin(missing_days)]
        )
        combined = pa.concat_tables([existing_table, new_table])
        pq.write_table(combined, month_output, compression="snappy")
        total_ticks = combined.num_rows
        del existing_table, new_table, combined
    else:
        month_df.to_parquet(month_output, index=False, engine="pyarrow", compression="snappy")
        total_ticks = len(month_df)

    file_size_mb = month_output.stat().st_size / (1024 ** 2)

    if verbose:
        print(
            f"   💾 Guardado: {month_output.name} "
            f"({total_ticks:,} ticks, {file_size_mb:.1f} MB)"
        )

    return {
        "downloaded": downloaded,
        "failed": failed,
        "file": month_output,
        "ticks": total_ticks,
    }


def download_year_ticks(
    symbol: str,
    year: int,
    output_dir: Path,
    verbose: bool = True,
) -> dict:
    """Descarga todos los meses de un año."""
    normalizer = DataNormalizer()

    results = []
    total_downloaded = 0
    total_failed = 0

    for month in range(1, 13):
        result = download_month_ticks(
            symbol=symbol,
            year=year,
            month=month,
            output_dir=output_dir,
            normalizer=normalizer,
            verbose=verbose,
        )
        results.append(result)
        total_downloaded += result["downloaded"]
        total_failed += result["failed"]
        gc.collect()

    return {
        "symbol": symbol,
        "year": year,
        "downloaded": total_downloaded,
        "failed": total_failed,
        "months": results,
    }


def print_summary(summaries: list[dict]):
    if not summaries:
        return

    print("\n" + "=" * 80)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 80)
    header = f"{'Symbol':<10} {'Year':>6} {'Downloaded':>12} {'Failed':>10}"
    print(header)
    print("-" * 80)

    for s in summaries:
        print(
            f"{s['symbol']:<10} "
            f"{s['year']:>6} "
            f"{s['downloaded']:>12} "
            f"{s['failed']:>10}"
        )

    total_dl = sum(s['downloaded'] for s in summaries)
    total_fail = sum(s['failed'] for s in summaries)
    print("-" * 80)
    print(f"{'TOTAL':<10} {'':<6} {total_dl:>12} {total_fail:>10}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Download raw ticks to parquet")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--month", type=int, default=None, choices=range(1, 13),
                        help="Specific month (1-12) to download (requires --year)")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    # Determine symbols
    if args.symbols:
        symbols = args.symbols
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = DEFAULT_SYMBOLS

    # Validate --month requires --year
    if args.month is not None and args.year is None:
        print("Error: --month requires --year")
        sys.exit(1)

    # Determine years
    if args.year is not None:
        years = [args.year]
    elif args.start_year is not None and args.end_year is not None:
        years = list(range(args.start_year, args.end_year + 1))
    else:
        print("Error: Specify --year or --start-year/--end-year")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    print("\n" + "📥" * 20)
    print(f"  RAW TICKS DOWNLOAD")
    print(f"  Symbols : {', '.join(symbols)}")
    print(f"  Years   : {years[0]} → {years[-1]} ({len(years)} years)")
    if args.month is not None:
        print(f"  Month   : {args.month:02d}")
    print(f"  Output  : {output_dir}")
    print("📥" * 20 + "\n")

    if args.dry_run:
        print("🔍 Dry run - would download:")
        for symbol in symbols:
            for year in years:
                months = [args.month] if args.month is not None else range(1, 13)
                for month in months:
                    month_output = output_dir / str(year) / f"{symbol}_{year}_{month:02d}.parquet"
                    target_days = get_target_days(year, month)
                    existing_dates = get_existing_dates(month_output)
                    missing = [d for d in target_days if d not in existing_dates]

                    if not target_days:
                        print(f"   {symbol} {year}-{month:02d}: 0 days (future)")
                        continue

                    print(
                        f"   {symbol} {year}-{month:02d}: "
                        f"{len(missing)}/{len(target_days)} missing"
                    )
        return

    summaries = []

    if args.month is not None:
        for symbol in symbols:
            for year in years:
                normalizer = DataNormalizer()
                month_result = download_month_ticks(
                    symbol=symbol,
                    year=year,
                    month=args.month,
                    output_dir=output_dir,
                    normalizer=normalizer,
                    verbose=True,
                )
                summaries.append({
                    "symbol": symbol,
                    "year": year,
                    "downloaded": month_result["downloaded"],
                    "failed": month_result["failed"],
                    "months": [month_result],
                })
                gc.collect()
    else:
        for symbol in symbols:
            for year in years:
                result = download_year_ticks(
                    symbol=symbol,
                    year=year,
                    output_dir=output_dir,
                    verbose=True,
                )
                summaries.append(result)

    print_summary(summaries)
    print("\n✅ Download complete!\n")


if __name__ == "__main__":
    main()