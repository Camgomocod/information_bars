"""Run a small end-to-end DRB and feature-engineering example.

The example intentionally uses generated ticks so a new contributor can verify
the installation without Binance access, a database, or Optuna studies.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.bars.info_bars import OptimizedInfoRunBars
from src.features.base_features import compute_all_features


def make_ticks(n_ticks: int = 300_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_ticks, freq="100ms")
    prices = 40_000 + np.cumsum(rng.normal(0, 2.0, n_ticks))
    quantities = rng.lognormal(mean=-0.1, sigma=0.35, size=n_ticks)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices.astype(np.float64),
            "quantity": quantities.astype(np.float64),
        }
    )


def main() -> int:
    ticks = make_ticks()
    bars = OptimizedInfoRunBars(save_path=None).get_drbs(
        ticks,
        exp_lambda=0.9975,
        init_exp_T=400,
    )
    if bars.empty:
        raise RuntimeError("Synthetic data did not produce any Dollar Run Bars")

    features = compute_all_features(
        bars,
        drop_warmup=True,
        winsorize=False,
        positioning_data=None,
        symbol=None,
    )
    output = Path("data_optimized/examples/synthetic_drb_features.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)
    print(f"Generated {len(bars):,} DRBs and {len(features):,} feature rows")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
