"""
Download Pipeline and Conversion to Dollar Bars with Visualization
Downloads tick data per day, converts to DRBs and DIBs, and generates visualizations
"""

import io
import os
import time
import zipfile
import requests
import pandas as pd
import warnings
from pathlib import Path
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ==================== CONFIGURATION ====================
BASE_URL = "https://data.binance.vision/data/spot/daily/trades"
SYMBOL = "BTCUSDT"
BASE_DIR = Path("data_comparison")
RAW_DIR = BASE_DIR / "raw_ticks"
BARS_DIR = BASE_DIR / "bars"
DRB_DIR = BARS_DIR / "dollar_run"
DIB_DIR = BARS_DIR / "dollar_imbalance"
PLOTS_DIR = BASE_DIR / "plots"

# Create directory structure
RAW_DIR.mkdir(parents=True, exist_ok=True)
DRB_DIR.mkdir(parents=True, exist_ok=True)
DIB_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Expected Binance columns
EXPECTED_COLUMNS = [
    "id",
    "price",
    "qty",
    "quote_qty",
    "time",
    "is_buyer_maker",
    "is_best_match",
]

# ==================== DOWNLOAD FUNCTIONS ====================


class DownloadData:
    def __init__(self, symbol="BTCUSDT", source="binance", data_dir="data_raw"):
        self.symbol = symbol
        self.source = source  # "binance", "local", o "auto" (local + fallback a binance)
        self.data_dir = Path(data_dir)

    def load_from_parquet(self, date_str):
        """
        Carga datos desde parquet local.
        date_str format: YYYY-MM-DD
        Optimizado: usa PyArrow predicate pushdown para cargar solo el día necesario.
        """
        year = date_str[:4]
        month = int(date_str[5:7])

        parquet_file = self.data_dir / year / f"{self.symbol}_{year}_{month:02d}.parquet"

        if not parquet_file.exists():
            return None

        try:
            import pyarrow.parquet as pq
            table = pq.read_table(
                parquet_file,
                filters=[('download_date', '==', date_str)]
            )
            df = table.to_pandas()

            if df.empty:
                return None

            df = df.rename(columns={
                "qty": "quantity",
                "quote_qty": "dollar_value",
                "datetime": "timestamp",
            })

            return df

        except Exception as e:
            print(f"❌ Error loading parquet: {e}")
            return None

    def read_binance_csv(self, file_obj):
        """Reads a Binance CSV from a file object"""
        try:
            content = file_obj.read()

            if not content or len(content) == 0:
                return None

            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                text_content = content.decode("latin-1")

            text_content = text_content.replace("\ufeff", "")

            if not text_content.strip():
                return None

            lines = text_content.strip().split("\n")
            if not lines:
                return None

            first_line = lines[0].strip()
            has_header = "id" in first_line.lower() and "price" in first_line.lower()

            from io import StringIO

            dtype_dict = {
                "id": "int64",
                "price": "float32",
                "qty": "float32",
                "quote_qty": "float32",
                "time": "int64",
                "is_buyer_maker": "bool",
                "is_best_match": "bool",
            }

            if has_header:
                df = pd.read_csv(
                    StringIO(text_content), dtype=dtype_dict, skipinitialspace=True
                )
            else:
                df = pd.read_csv(
                    StringIO(text_content),
                    header=None,
                    names=EXPECTED_COLUMNS,
                    dtype=dtype_dict,
                    skipinitialspace=True,
                )

            if df.empty:
                return None

            for bool_col in ["is_buyer_maker", "is_best_match"]:
                if bool_col in df.columns and df[bool_col].dtype == "object":
                    df[bool_col] = df[bool_col].astype(str).str.strip().str.lower()
                    df[bool_col] = df[bool_col].map({"true": True, "false": False})

            if df["time"].max() > 1e14:
                df["datetime"] = pd.to_datetime(df["time"], unit="us")
            else:
                df["datetime"] = pd.to_datetime(df["time"], unit="ms")

            df["date"] = df["datetime"].dt.date

            df = df.rename(
                columns={
                    "qty": "quantity",
                    "quote_qty": "dollar_value",
                    "datetime": "timestamp",
                }
            )

            df = df.dropna(subset=["timestamp", "time"])

            if df.empty:
                return None

            return df

        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return None

    def download_day(self, date_str):
        """
        Downloads data for a specific day.
        - source='local': solo carga desde parquet local
        - source='binance': solo descarga desde Binance
        - source='auto': intenta local primero, si no existe descarga desde Binance
        """
        if self.source in ("local", "auto"):
            print(f"📂 Loading {date_str} from local...", end=" ")
            df = self.load_from_parquet(date_str)
            if df is not None and not df.empty:
                print(f"✅ {len(df):,} ticks loaded")
                return df
            else:
                if self.source == "local":
                    print(f"❌ Not found in local files")
                    return None
                # Auto mode: fallback to Binance
                print(f"⚠️  Not in local, falling back to Binance...")

        # Download from Binance (with retry — data may not be published yet at 06:00 UTC)
        url = f"{BASE_URL}/{self.symbol}/{self.symbol}-trades-{date_str}.zip"

        # Retry delays are configurable via BINANCE_RETRY_DELAYS (comma-separated
        # seconds). Default [120,300,600] suits the daily inference cron (data may
        # not be published yet); batch/historical jobs can set BINANCE_RETRY_DELAYS=""
        # to fail fast on days that simply don't exist.
        retry_cfg = os.environ.get("BINANCE_RETRY_DELAYS")
        if retry_cfg is None:
            retry_delays = [120, 300, 600]
        else:
            retry_delays = [int(d) for d in retry_cfg.split(",") if d.strip()] or [0]

        for attempt, delay in enumerate(retry_delays):
            if attempt > 0:
                print(f"\n⏳ Retry {attempt}/{len(retry_delays)-1} in {delay}s...", end=" ")
                time.sleep(delay)

            if attempt == 0:
                print(f"📥 Downloading {date_str}...", end=" ")
            else:
                print(f"📥 Downloading {date_str} (attempt {attempt+1})...", end=" ")

            try:
                response = requests.get(url, timeout=30)

                if response.status_code == 404:
                    if attempt < len(retry_delays) - 1:
                        print(f"❌ Not found (will retry)")
                        continue
                    print("❌ Not found")
                    return None

                if response.status_code != 200:
                    if attempt < len(retry_delays) - 1:
                        print(f"❌ HTTP {response.status_code} (will retry)")
                        continue
                    print(f"❌ HTTP {response.status_code}")
                    return None

                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    csv_filename = zip_file.namelist()[0]

                    with zip_file.open(csv_filename) as csv_file:
                        df = self.read_binance_csv(csv_file)

                if df is None or df.empty:
                    print("❌ Empty or invalid CSV")
                    return None

                print(f"✅ {len(df):,} ticks downloaded")
                return df

            except Exception as e:
                if attempt < len(retry_delays) - 1:
                    print(f"❌ Error: {e} (will retry)")
                    continue
                print(f"❌ Error: {e}")
                return None

        return None
