import sys
import gc
import warnings
from pathlib import Path
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.connectors.download_data import DownloadData
from src.bars.info_bars import OptimizedInfoRunBars
from src.normalizers.data_normalizer import DataNormalizer

warnings.filterwarnings("ignore")

def process_history():
    # Configuration
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
    YEARS = [2021, 2022, 2023, 2024]

    # Optimal Hyperparameters per Symbol
    SYMBOL_CONFIG = {
        "BTCUSDT": {"exp_lambda": 0.995, "init_exp_T": 3200},
        "ETHUSDT": {"exp_lambda": 0.995, "init_exp_T": 3200},
        "BNBUSDT": {"exp_lambda": 0.995, "init_exp_T": 3200},
        "SOLUSDT": {"exp_lambda": 0.995, "init_exp_T": 3200},
    }

    # Directories
    BARS_DIR = project_root / "data" / "bars" / "dollar_run"
    BARS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("🚀 STARTING HISTORY PROCESSING PIPELINE (DAILY FILES)")
    print("="*60)
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Years: {YEARS}")
    print("Configurations:")
    for sym, cfg in SYMBOL_CONFIG.items():
        print(f"  - {sym}: Lambda={cfg['exp_lambda']}, T={cfg['init_exp_T']}")
    print(f"Output: {BARS_DIR}")
    print("="*60 + "\n")

    for symbol in SYMBOLS:
        # Get specific params for this symbol
        params = SYMBOL_CONFIG.get(symbol)
        if not params:
            print(f"⚠️ No config for {symbol}, skipping.")
            continue

        exp_lambda = params["exp_lambda"]
        init_exp_T = params["init_exp_T"]

        print(f"\n{'#'*60}")
        print(f"# PROCESSING: {symbol}")
        print(f"# Params: Lambda={exp_lambda}, T={init_exp_T}")
        print(f"{'#'*60}\n")

        symbol_dir = BARS_DIR / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        downloader = DownloadData(symbol=symbol)
        bar_generator = OptimizedInfoRunBars(symbol_dir) # Path not used in get_drbs but required by init
        normalizer = DataNormalizer()

        for year in YEARS:
            for month in range(1, 13):
                # Skip future months
                if year == date.today().year and month > date.today().month:
                    continue

                month_str = f"{year}-{month:02d}"
                print(f"📅 Processing {month_str}...")

                # Calculate start and end dates for the month
                start_date = date(year, month, 1)
                if month == 12:
                    end_date = date(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_date = date(year, month + 1, 1) - timedelta(days=1)

                # Limit to yesterday if current month
                if year == date.today().year and month == date.today().month:
                    end_date = min(end_date, date.today() - timedelta(days=1))

                current_date = start_date

                while current_date <= end_date:
                    date_str = current_date.strftime("%Y-%m-%d")
                    daily_output_file = symbol_dir / f"{symbol}_{date_str}_drbs.parquet"

                    if daily_output_file.exists():
                        # print(f"   ✅ {date_str} exists")
                        current_date += timedelta(days=1)
                        continue

                    try:
                        # 1. Download
                        df_raw = downloader.download_day(date_str)

                        if df_raw is not None and not df_raw.empty:
                            # 2. Clean
                            df_clean = normalizer.clean_raw_ticks(df_raw)

                            if not df_clean.empty:
                                # 3. Generate Bars
                                # Ensure required columns exist
                                if 'dollar_value' not in df_clean.columns:
                                    df_clean['dollar_value'] = df_clean['price'] * df_clean['quantity']

                                drbs = bar_generator.get_drbs(
                                    df_clean,
                                    exp_lambda=exp_lambda,
                                    init_exp_T=init_exp_T
                                )

                                if not drbs.empty:
                                    # Save DAILY file
                                    drbs.to_parquet(daily_output_file, index=False)
                                    print(f"   💾 {date_str}: {len(drbs)} bars")
                                else:
                                    print(f"   ⚠️ {date_str}: No bars generated")

                            # Clean up raw data immediately
                            del df_raw
                            del df_clean
                            gc.collect()
                        else:
                             print(f"   ⚠️ {date_str}: No raw data")

                    except Exception as e:
                        print(f"   ❌ Error on {date_str}: {e}")

                    current_date += timedelta(days=1)

    print("\n" + "🎉"*20)
    print("ALL DONE!")

if __name__ == "__main__":
    process_history()
