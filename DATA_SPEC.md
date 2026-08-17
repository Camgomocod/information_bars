# trading-core — Data Specification

> Technical documentation for the generated training datasets.
> This document serves as the authoritative reference for data structure, features, and metadata.

---

## 1. Dataset Overview

### Files Generated

| Year | Symbol | Filename | Shape | Size |
|------|--------|----------|-------|------|
| 2023 | BTCUSDT | `BTCUSDT_2023.parquet` | 71,380 × 32 | 10.4 MB |
| 2024 | BTCUSDT | `BTCUSDT_2024.parquet` | 64,234 × 32 | 9.4 MB |
| 2025 | BTCUSDT | `BTCUSDT_2025.parquet` | 47,385 × 32 | 7.0 MB |

### Data Source
- **Exchange**: Binance (spot)
- **Interval**: Dollar Run Bars (DRB) - information-driven sampling
- **Tick source**: `data_raw/{year}/BTCUSDT_{year}_{month:02d}.parquet`
- **Positioning source**: Binance Futures API — funding rate cached at `data_raw/futures/{year}/`

---

## 2. Column Reference

### 2.1 OHLCV Base Columns

| Column | Type | Description |
|--------|------|-------------|
| `open_time` | datetime64[ns] | Bar open timestamp |
| `close_time` | datetime64[ns] | Bar close timestamp |
| `open` | float32 | Opening price |
| `high` | float32 | Highest price in bar |
| `low` | float32 | Lowest price in bar |
| `close` | float32 | Closing price |
| `n_ticks` | int32 | Number of ticks in bar |
| `volume` | float32 | Trading volume |
| `dollar_value` | float32 | Total dollar volume |

### 2.2 DRB Hyperparameters

| Column | Type | Description |
|--------|------|-------------|
| `exp_lambda` | float32 | Exponential decay lambda (0.996 - 0.999) |
| `init_exp_T` | int32 | Initial tick threshold |

> **Note**: These params vary by month due to walk-forward optimization.

### 2.3 Feature Engineered Columns

| Column | Type | Description |
|--------|------|-------------|
| `log_return` | float32 | Log return: ln(close_t / close_{t-1}) |
| `frac_diff_return` | float32 | Fractionally differenced log-price (d=0.4, configurable) |
| `rolling_volatility` | float32 | Rolling 20-bar std of log returns |
| `bar_range` | float32 | (high - low) / open |
| `atr_pct` | float32 | ATR as % of price |
| `volume_z` | float32 | Z-score of volume vs 20-bar rolling window |
| `dollar_value_z` | float32 | Z-score of dollar_value |
| `n_ticks_z` | float32 | Z-score of n_ticks |
| `vwap` | float32 | Volume Weighted Average Price |
| `price_to_vwap` | float32 | (close - vwap) / vwap |
| `bar_duration_secs` | float32 | Bar duration in seconds |
| `rsi` | float32 | RSI(14) |
| `bb_pct_b` | float32 | Bollinger Bands %B |
| `macd_hist` | float32 | MACD histogram |

### 2.4 Market Positioning Columns

| Column | Type | Description |
|--------|------|-------------|
| `funding_rate_mean` | float32 | Rolling mean of last 3 funding rate periods (≈24h). Positive extreme = over-leveraged long (contrarian signal) |

> **Note**: `funding_rate_mean` may contain NaN for bars that precede the first available funding rate data. It is merged via `merge_asof(direction='backward')` to prevent lookahead bias.

### 2.5 Metadata Columns

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | category | Trading pair (BTCUSDT) |
| `year` | int16 | Year (2023, 2024) |
| `month` | int8 | Month (1-12) |
| `study_source` | category | Optuna study name (e.g., `bayesian_2022_12_w2m`) |
| `completion_rate` | float32 | Study completion rate (0.0-1.0) |
| `failed_day` | int8 | 1 if bar belongs to a failed day, 0 otherwise |
| `sample_weight` | float32 | 1.0 for normal, 0.5 for failed bars |

---

## 3. Feature Details

### 3.1 Returns & Volatility

```python
log_return = ln(close_t / close_{t-1})

# Fractional differentiation (d=0.4 by default, configurable via features_config)
# Preserves memory while achieving stationarity
frac_diff_return = ReturnsCalculator.compute_frac_diff(log_close, d=0.4)

# Rolling volatility (20-bar window) = sqrt(MA(log_return^2, window))
rolling_volatility = sqrt(MA(log_return^2, window=20))
```

### 3.2 Price-Based Features

```python
bar_range = (high - low) / open

# ATR as percentage
atr_pct = ATR(close, high, low, period=14) / close

# VWAP
vwap = cumsum(price * volume) / cumsum(volume)
price_to_vwap = (close - vwap) / vwap
```

### 3.3 Volume-Based Z-Scores

```python
# Z-score normalization: (value - rolling_mean) / rolling_std
volume_z = (volume - rolling_mean(volume, 20)) / rolling_std(volume, 20)
dollar_value_z = (dollar_value - rolling_mean(...)) / rolling_std(...)
n_ticks_z = (n_ticks - rolling_mean(...)) / rolling_std(...)
```

### 3.4 Technical Indicators

```python
# RSI(14)
rsi = RSI(close, period=14)

# Bollinger Bands %B
bb_pct_b = (close - lower_band) / (upper_band - lower_band)

# MACD Histogram
macd, signal, hist = MACD(close, fast=12, slow=26, signal=9)
macd_hist = hist
```

### 3.5 Market Positioning Features

Sourced from Binance Futures API (public, no API key required). Merged via
`merge_asof(direction='backward')` to prevent lookahead bias.

```python
# Funding Rate: 8h intervals, rolling mean of last 3 periods (~24h)
# Source: /fapi/v1/fundingRate
funding_rate_mean = rolling_mean(funding_rate, window=3)
```

---

## 4. Walk-Forward Optimization Context

### Study Naming Convention

```
bayesian_YYYY_MM_w2m
```

- Covers 2 months: (YYYY-MM-1) and (YYYY-MM)
- Used for: (YYYY-MM+1) and (YYYY-MM+2)

### Parameter Evolution (BTCUSDT)

| Month | exp_lambda | init_exp_T | Study |
|-------|------------|------------|-------|
| 2023-01 | 0.9975 | 14,738 | bayesian_2022_12_w2m |
| 2023-06 | 0.9989 | 6,057 | bayesian_2023_04_w2m |
| 2023-12 | 0.9964 | 22,448 | bayesian_2023_10_w2m |
| 2024-06 | 0.9990 | 3,884 | bayesian_2024_04_w2m |
| 2024-12 | 0.9990 | 8,369 | bayesian_2024_10_w2m |

### Failed Day Handling

- **Definition**: Day with < 50 bars generated
- **Recovery**: Separate optimization with custom params in `experiments/failed_days_YYYY_MM/`
- **Sample weighting**: Failed bars get 0.5 weight (down-weighted in training)

---

## 5. Quality Metrics

### 2023 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total bars | 71,380 |
| Failed bars | 4,580 (6.4%) |
| Mean bar duration | 371 sec (~6 min) |
| Mean n_ticks | 11,997 |
| Price range | $16,553 - $44,631 |

### 2024 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total bars | 64,234 |
| Failed bars | 930 (1.4%) |
| Mean bar duration | 503 sec (~8 min) |
| Mean n_ticks | 14,725 |
| Price range | $38,600 - $108,194 |

### 2025 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total bars | 47,385 |
| Failed bars | 1,422 (3.0%) |
| Mean bar duration | 706 sec (~12 min) |
| Mean n_ticks | 26,983 |
| Price range | $74,551 - $126,195 |

---

## 6. Usage Examples

### 6.1 Loading Data

```python
import pandas as pd

df = pd.read_parquet('data_optimized/training/2023/BTCUSDT_2023.parquet')
print(f"Loaded {len(df):,} bars")
```

### 6.2 Filtering by Weight

```python
# Use all bars with sample weights
df_weighted = df.copy()
df_weighted['weight'] = df_weighted['sample_weight']

# Or exclude failed bars entirely
df_clean = df[df['failed_day'] == 0]
```

### 6.3 Feature Engineering

```python
# Select features for ML
feature_cols = [
    # Bar-level features
    'log_return', 'frac_diff_return', 'rolling_volatility',
    'bar_range', 'atr_pct', 'volume_z', 'dollar_value_z',
    'n_ticks_z', 'price_to_vwap', 'rsi', 'bb_pct_b', 'macd_hist',
    # Positioning feature (may have NaN in early bars)
    'funding_rate_mean',
]

X = df[feature_cols]
# Handle funding_rate_mean NaN: fill with 0 or use models that support missing values
X = X.fillna(0)
y = (df['log_return'].shift(-1) > 0).astype(int)  # Next-bar direction
```

### 6.4 Walk-Forward Validation

```python
# Train on 2023, test on 2024
train = df[df['year'] == 2023]
test = df[df['year'] == 2024]
```

---

## 7. Schema Validation

```python
import pandera as pa
from pandera import Column, Check, DataFrameSchema

schema = DataFrameSchema({
    "open_time": Column(pa.DateTime, nullable=False),
    "close_time": Column(pa.DateTime, nullable=False),
    "open": Column(pa.Float, nullable=False, gt=0),
    "high": Column(pa.Float, nullable=False, gt=0),
    "low": Column(pa.Float, nullable=False, gt=0),
    "close": Column(pa.Float, nullable=False, gt=0),
    "n_ticks": Column(pa.Int, nullable=False, ge=1),
    "volume": Column(pa.Float, nullable=False, ge=0),
    "dollar_value": Column(pa.Float, nullable=False, ge=0),
    "exp_lambda": Column(pa.Float, nullable=False, ge=0.99, le=1.0),
    "init_exp_T": Column(pa.Int, nullable=False, ge=1),
    "log_return": Column(pa.Float, nullable=True),
    "frac_diff_return": Column(pa.Float, nullable=True),
    "rolling_volatility": Column(pa.Float, nullable=True),
    "bar_range": Column(pa.Float, nullable=True, ge=0),
    "atr_pct": Column(pa.Float, nullable=True, ge=0),
    "volume_z": Column(pa.Float, nullable=True),
    "dollar_value_z": Column(pa.Float, nullable=True),
    "n_ticks_z": Column(pa.Float, nullable=True),
    "vwap": Column(pa.Float, nullable=True, gt=0),
    "price_to_vwap": Column(pa.Float, nullable=True),
    "bar_duration_secs": Column(pa.Float, nullable=True, ge=0),
    "rsi": Column(pa.Float, nullable=True, ge=0, le=100),
    "bb_pct_b": Column(pa.Float, nullable=True, ge=0, le=1),
    "macd_hist": Column(pa.Float, nullable=True),
    "funding_rate_mean": Column(pa.Float, nullable=True),
    "symbol": Column(pa.String, nullable=False),
    "year": Column(pa.Int, nullable=False),
    "month": Column(pa.Int, nullable=False, ge=1, le=12),
    "study_source": Column(pa.String, nullable=False),
    "completion_rate": Column(pa.Float, nullable=False, ge=0, le=1),
    "failed_day": Column(pa.Int, nullable=False, ge=0, le=1),
    "sample_weight": Column(pa.Float, nullable=False, ge=0, le=1),
})
```

---

## 8. Reproducibility

### Generating New Data

```bash
# Full year with local data
micromamba run -n trading-core python scripts/build_yearly_training_data.py \
    --source local --symbol BTCUSDT --year 2023

# Run optimization first if needed
micromamba run -n trading-core python scripts/run_year_optimization.py \
    --year 2023 --trials 50
```

### Dependencies

```yaml
# environment.yml
dependencies:
  - python=3.10
  - pandas
  - numpy
  - pyarrow
  - numba
  - vectorbt
  - optuna
  - scipy
  - scikit-learn
  - pyyaml
```

---

## 9. Citation

If you use this dataset in your research, please cite:

```bibtex
@software{information-bars,
  title = {Information Bars: Crypto Market Microstructure Pipeline},
  author = {Goyes, Camilo},
  year = {2024},
  url = {https://github.com/Camgomocod/information_bars}
}
```

---

*Generated: 2024-04-08*
*Pipeline version: 2.0*
