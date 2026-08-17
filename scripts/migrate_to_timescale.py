#!/usr/bin/env python3
"""
migrate_to_timescale.py — Migración única desde parquets + experimentos a TimescaleDB

Uso:
    micromamba run -n trading-core python scripts/migrate_to_timescale.py

Migración de:
    - data_optimized/training/20*/*.parquet  → tabla bars (hypertable)
    - experiments/bayesian_*_w2m/*/*.pkl     → tabla studies
    - experiments/bayesian_*_w2m/*/*_failed_days_analysis.txt → tabla failed_days

Tiempo estimado: ~30 minutos
"""

import sys
from pathlib import Path
from datetime import datetime
import re
import json

import pandas as pd
import joblib
import yaml

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage.timescale_client import TimescaleDBClient
from src.storage.db_config import get_db_url

DB_URL = get_db_url()
EXPERIMENTS_DIR = Path("experiments")
TRAINING_DIR = Path("data_optimized/training")


def extract_period_from_window(window_name: str) -> str:
    """
    'bayesian_2024_03_w2m' -> '2024-03'
    """
    m = re.match(r'bayesian_(\d{4})_(\d{2})_w2m', window_name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return window_name


def migrate_bars(db: TimescaleDBClient):
    """
    Migrar todos los parquets de training a tabla bars.
    """
    print("\n" + "=" * 60)
    print("MIGRANDO BARS (parquets → hypertable)")
    print("=" * 60)

    total_rows = 0
    for year_dir in sorted(TRAINING_DIR.glob("20*")):
        year = year_dir.name
        for parquet_file in sorted(year_dir.glob("*.parquet")):
            symbol = parquet_file.name.split('_')[0]
            print(f"\n  📂 {parquet_file.name} ({parquet_file.stat().st_size / 1024 / 1024:.1f} MB)")

            df = pd.read_parquet(parquet_file)
            n = db.write_bars(df, symbol=symbol)
            total_rows += n
            print(f"     ✅ {n:,} rows inserted")

    print(f"\n  TOTAL: {total_rows:,} bars migrated")


def migrate_studies(db: TimescaleDBClient):
    """
    Migrar todos los estudios de experiments/ a tabla studies.
    """
    print("\n" + "=" * 60)
    print("MIGRANDO STUDIES (pkl + yaml → tabla studies)")
    print("=" * 60)

    total = 0
    for study_dir in sorted(EXPERIMENTS_DIR.glob("bayesian_*_w2m")):
        window_name = study_dir.name
        period = extract_period_from_window(window_name)

        for symbol_dir in sorted(study_dir.glob("*")):
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name

            report_file = symbol_dir / f"{symbol}_bayesian_report.txt"
            bounds_file = symbol_dir / f"{symbol}_search_bounds.yaml"

            if not report_file.exists():
                continue

            try:
                db.upsert_study_from_report(
                    symbol=symbol,
                    period=period,
                    report_path=report_file,
                    bounds_path=bounds_file if bounds_file.exists() else None,
                )
                total += 1
                print(f"  ✅ {window_name}/{symbol}")
            except Exception as e:
                print(f"  ⚠️  {window_name}/{symbol}: {e}")

    print(f"\n  TOTAL: {total} studies migrated")


def migrate_failed_days(db: TimescaleDBClient):
    """
    Migrar failed_days_analysis a tabla failed_days.
    """
    print("\n" + "=" * 60)
    print("MIGRANDO FAILED DAYS (txt → tabla failed_days)")
    print("=" * 60)

    total = 0
    for study_dir in sorted(EXPERIMENTS_DIR.glob("bayesian_*_w2m")):
        window_name = study_dir.name
        period = extract_period_from_window(window_name)

        for symbol_dir in sorted(study_dir.glob("*")):
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name

            analysis_file = symbol_dir / f"{symbol}_failed_days_analysis.txt"
            if not analysis_file.exists():
                continue

            try:
                db.insert_failed_days(symbol, period, analysis_file)
                total += 1
                print(f"  ✅ {window_name}/{symbol}")
            except Exception as e:
                print(f"  ⚠️  {window_name}/{symbol}: {e}")

    print(f"\n  TOTAL: {total} failed_days entries migrated")


def main():
    print(f"\n{'=' * 60}")
    print(f"  MIGRACIÓN A TIMESCALEDB")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    db = TimescaleDBClient(DB_URL)

    try:
        migrate_bars(db)
        migrate_studies(db)
        migrate_failed_days(db)

        print("\n" + "=" * 60)
        print("✅ MIGRACIÓN COMPLETADA")
        print(f"   Final: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Migración interrumpida por usuario")
    finally:
        db.close()


if __name__ == "__main__":
    main()
