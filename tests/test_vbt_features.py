import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings("ignore")

SYMBOL = "BTCUSDT"
BARS_DIR = project_root / "data" / "bars" / "dollar_run" / SYMBOL


def _load_bars() -> pd.DataFrame | None:
    files = sorted(BARS_DIR.glob(f"{SYMBOL}_2024-12-*_drbs.parquet"))
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df


def test_data_files_exist():
    if not BARS_DIR.exists():
        pytest.skip(f"DRB data directory {BARS_DIR} does not exist — run process_history_to_bars.py first")
    files = sorted(BARS_DIR.glob(f"{SYMBOL}_2024-12-*_drbs.parquet"))
    if len(files) == 0:
        pytest.skip(f"No daily DRB parquet files found for {SYMBOL} in Dec 2024")


def test_feature_generation():
    df = _load_bars()
    if df is None:
        pytest.skip("No DRB parquet data available — run process_history_to_bars.py first")

    assert len(df) > 0, "Loaded empty DataFrame"
    assert "close" in df.columns, "Missing 'close' column"

    close_price = df["close"]

    rsi = vbt.RSI.run(close_price, window=14)
    bbands = vbt.BBANDS.run(close_price, window=20, alpha=2)
    macd = vbt.MACD.run(close_price, fast_window=12, slow_window=26, signal_window=9)

    features = pd.DataFrame(
        {
            "close": close_price,
            "rsi": rsi.rsi,
            "bb_upper": bbands.upper,
            "bb_middle": bbands.middle,
            "bb_lower": bbands.lower,
            "macd": macd.macd,
            "macd_signal": macd.signal,
        }
    )
    features["log_return"] = np.log(close_price / close_price.shift(1))
    features["volatility"] = features["log_return"].rolling(window=20).std()

    assert not features.empty, "Features DataFrame is empty"
    assert "rsi" in features.columns
    assert "macd" in features.columns
    assert "log_return" in features.columns
    assert "volatility" in features.columns
    assert features["rsi"].between(0, 100).all(), "RSI out of [0, 100] range"
    assert features["log_return"].notna().sum() > 0, "All log_returns are NaN"
    assert features["volatility"].notna().sum() > 0, "All volatility values are NaN"


if __name__ == "__main__":
    test_data_files_exist()
    test_feature_generation()
    print("✅ All checks passed")
