"""
Data Normalizer Module
Responsible for cleaning and preprocessing financial data.
Includes:
1. Tick Cleaning (Outliers, MAD Filter)
2. Missing values handling
3. Fractional Differentiation (FracDiff)
"""

import numpy as np
import pandas as pd


class DataNormalizer:
    """
    Utility class for financial data normalization.
    """

    @staticmethod
    def clean_raw_ticks(
        df: pd.DataFrame,
        mad_window: int = None,
        k: float = None,
        audit: object = None,
    ) -> pd.DataFrame:
        """
        Cleans anomalous ticks from a raw data DataFrame.

        Applied filters:
        1. Price > 0 and Quantity > 0
        2. Outlier Filter based on MAD (Median Absolute Deviation), strictly causal

        Causality: rolling stats use ``center=False`` and zero-MAD gaps are
        forward-filled ONLY (ffill). Legacy code used bfill().ffill(), which
        pulled future rows into the warm-up region (lookahead). With ffill-only,
        a tick is judged solely against its past.

        Args:
            df: DataFrame with ['price', 'quantity'] columns
            mad_window: Window size for moving median calculation
            k: Number of deviations to be considered an outlier
            audit: optional TickAudit to accumulate 'ticks_mad_removed' into

        Returns:
            Clean DataFrame
        """
        if mad_window is None or k is None:
            try:
                from src.config import get_cleaning_config

                cfg = get_cleaning_config()
                if mad_window is None:
                    mad_window = cfg.get("mad_window", 100)
                if k is None:
                    k = cfg.get("k_factor", 10.0)
            except Exception:
                mad_window = mad_window or 100
                k = k or 10.0
        if df is None or df.empty:
            return df

        # 1. Basic Sanity Filter
        # Remove negative or zero prices/quantities
        mask_sanity = (df["price"] > 0) & (df["quantity"] > 0)
        n_sanity_removed = (~mask_sanity).sum()

        if n_sanity_removed > 0:
            print(f"   🧹 Sanity check: Removed {n_sanity_removed} invalid ticks (price/qty <= 0)")

        df_clean = df[mask_sanity].copy()

        if df_clean.empty:
            return df_clean

        # 2. MAD Filter (Median Absolute Deviation)
        # Unidirectional rolling window (center=False): each tick is compared
        # only against its PAST. Warm-up rows (first mad_window) yield NaN MAD
        # and are kept (fillna(0) → mod_z=0, not an outlier).

        rolling_median = df_clean["price"].rolling(window=mad_window, center=False).median()

        deviation = np.abs(df_clean["price"] - rolling_median)

        rolling_mad = deviation.rolling(window=mad_window, center=False).median()

        # Avoid division by zero — ffill ONLY (causal). No bfill: backfilling
        # would leak future deviations into past rows.
        rolling_mad = rolling_mad.replace(0, np.nan).ffill()

        mod_z_score = 0.6745 * deviation / rolling_mad

        mask_outlier = mod_z_score.fillna(0) > k

        n_outliers = int(mask_outlier.sum())
        if n_outliers > 0:
            print(f"   🧹 Outlier filter: Removed {n_outliers} ticks (MAD > {k})")
        if audit is not None:
            audit.ticks_mad_removed += n_outliers

        return df_clean[~mask_outlier]

    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values in a bars DataFrame.

        Strategy:
        - Prices (Open, High, Low, Close): Forward Fill (price remains constant)
        - Volume/Dollar Value: Fill with 0 (no activity)
        - Ticks: Fill with 0

        Args:
            df: Bars DataFrame

        Returns:
            DataFrame without NaNs
        """
        if df is None or df.empty:
            return df

        df_clean = df.copy()

        # Price columns
        price_cols = [c for c in df_clean.columns if c in ["open", "high", "low", "close", "vwap"]]
        if price_cols:
            df_clean[price_cols] = df_clean[price_cols].ffill()

        # Volume/Activity columns
        vol_cols = [
            c for c in df_clean.columns if c in ["volume", "dollar_value", "n_ticks", "buy_volume", "sell_volume"]
        ]
        if vol_cols:
            df_clean[vol_cols] = df_clean[vol_cols].fillna(0)

        # Final cleanup for edges (e.g. beginning of DF)
        df_clean = df_clean.dropna()

        return df_clean

    @staticmethod
    def get_weights_ffd(d, thres, lim):
        """
        Calculates weights for fractional differentiation (Fixed Window).
        Helper function.
        """
        w, k = [1.0], 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < thres:
                break
            w.append(w_k)
            k += 1
            if k >= lim:
                break
        return np.array(w[::-1]).reshape(-1, 1)

    @staticmethod
    def frac_diff_fixed(series, d, thres=1e-5, window=None):
        """
        Applies Fractional Differentiation (Fixed Window) to a series.

        Args:
            series: Pandas Series (e.g., log-prices)
            d: Fractional order (float, e.g., 0.4)
            thres: Weight cutoff threshold for memory pruning
            window: Optional fixed window (if None, bounded by thres)

        Returns:
            Fractionally differentiated Pandas Series
        """
        # 1. Calc weights
        # Window limit to prevent infinite loop for low thresholds
        lim = len(series) if window is None else window
        w = DataNormalizer.get_weights_ffd(d, thres, lim)
        width = len(w) - 1

        # 2. Apply weights (convolution)
        # Utilizing numpy for efficiency
        # Padded NaNs initially due to insufficient history

        df = pd.DataFrame(series)

        shifted = {f"shift_{k}": df.iloc[:, 0].shift(k) for k in range(width + 1)}
        df_temp = pd.DataFrame(shifted).dropna()

        if df_temp.empty:
            return pd.Series(index=series.index, dtype=float)

        # Perform dot product
        result = np.dot(df_temp.values, w)

        return pd.Series(result.flatten(), index=df_temp.index, name=series.name)

    @staticmethod
    def winsorize(series: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
        """
        Applies Winsorization (clipping) to a series to limit extreme outliers.
        Highly useful for ML features like returns or volatility before modeling.

        WARNING: This computes quantiles over the FULL series. When used in a
        time-series pipeline it leaks future information into past rows. Prefer
        fitting a Winsorizer on train data only and transforming out-of-sample
        rows with the fitted limits (see Winsorizer).

        Args:
            series: Pandas Series
            limits: Tuple (lower_percentile, upper_percentile). E.g: (0.01, 0.01) clips 1% on both tails.

        Returns:
            Series with extreme tails clipped to boundary limits.
        """
        if series is None or series.empty:
            return series

        lower = series.quantile(limits[0])
        upper = series.quantile(1 - limits[1])
        return series.clip(lower=lower, upper=upper)

    @staticmethod
    def to_log_prices(df: pd.DataFrame, columns: list = ["close"]) -> pd.DataFrame:
        """
        Converts price columns into logarithmic prices.
        Crucial step symmetric statistical properties prior to calculating Returns or FracDiff.

        Args:
            df: Bars DataFrame
            columns: List of columns to convert

        Returns:
            DataFrame appended with 'log_{col}' columns
        """
        df_log = df.copy()
        for col in columns:
            if col in df_log.columns:
                # Prevent log(0) and log(negatives)
                mask = df_log[col] > 0
                if mask.all():
                    df_log[f"log_{col}"] = np.log(df_log[col])
                else:
                    # Si hay ceros/negativos, advertir y calcular solo donde sea válido
                    print(f"⚠️  Warning: Non-positive values in {col}, computing log only for positive values.")
                    df_log.loc[mask, f"log_{col}"] = np.log(df_log.loc[mask, col])
        return df_log


class Winsorizer:
    """
    Fit/transform winsorizer that prevents lookahead leakage.

    Quantiles are computed ONCE on the training split (fit) and then the same
    fixed limits are applied to any later rows (transform). This guarantees that
    modifying future data cannot change how past rows are clipped.

    Usage:
        wz = Winsorizer(limits=(0.01, 0.01))
        train_cols = wz.fit_transform(train[cols])
        test_cols  = wz.transform(test[cols])     # same limits as train
    """

    def __init__(self, limits: tuple = (0.01, 0.01)):
        self.limits = limits
        self.bounds: dict[str, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> "Winsorizer":
        """Compute per-column clip bounds from df (training split only)."""
        if df is None or df.empty:
            return self
        for col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().sum() == 0:
                continue
            lower = float(series.quantile(self.limits[0]))
            upper = float(series.quantile(1 - self.limits[1]))
            self.bounds[col] = (lower, upper)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted bounds to df. Columns without fitted bounds are unchanged."""
        out = df.copy()
        for col, (lower, upper) in self.bounds.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=lower, upper=upper)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @staticmethod
    def from_fitted(bounds: dict[str, tuple[float, float]], limits: tuple = (0.01, 0.01)) -> "Winsorizer":
        wz = Winsorizer(limits=limits)
        wz.bounds = dict(bounds)
        return wz


class AccumulatingWinsorizer:
    """
    Causally-accumulating winsorizer for streaming / month-by-month pipelines.

    Unlike ``Winsorizer`` (fit once on train), this keeps a bounded reservoir
    of PAST feature values per column. ``transform`` clips a new batch using
    quantiles of the reservoir ONLY (strictly-prior data — no lookahead), and
    ``update`` absorbs the just-processed batch into the reservoir afterwards.

    Rule enforced across the pipeline:
        limits(month M) = strictly-prior months only
        update(M) happens AFTER transform(M)

    The reservoir is bounded (``max_samples`` per column) using deterministic
    reservoir sampling, so memory is O(columns × max_samples) regardless of how
    many months have been ingested.

    Usage:
        wz = AccumulatingWinsorizer(limits=(0.01, 0.01), max_samples=20000)
        clipped = wz.transform(new_features)   # uses only past data
        wz.update(new_features)                # absorb for future months
    """

    def __init__(self, limits: tuple = (0.01, 0.01), max_samples: int = 20000, seed: int = 0):
        self.limits = limits
        self.max_samples = int(max_samples)
        self._rng = np.random.default_rng(seed)
        self._reservoirs: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}

    # ── reservoir management ─────────────────────────────────────────────

    def _absorb_column(self, col: str, values: np.ndarray) -> None:
        if values.size == 0:
            return
        count = self._counts.get(col, 0)
        arr = self._reservoirs.get(col)
        if arr is None:
            cap = min(values.size, self.max_samples)
            self._reservoirs[col] = values[:cap].astype(np.float64)
            self._counts[col] = values.size
            return
        if arr.size + values.size <= self.max_samples:
            self._reservoirs[col] = np.concatenate([arr, values]).astype(np.float64)
            self._counts[col] = count + values.size
            return
        # Reservoir sampling: keep the existing reservoir, probabilistically
        # replace slots with new values.
        total = count
        new_arr = arr.copy()
        # Number of new samples expected to land in the reservoir.
        for v in values:
            total += 1
            j = int(self._rng.integers(0, total))
            if j < new_arr.size:
                new_arr[j] = float(v)
        self._reservoirs[col] = new_arr
        self._counts[col] = total

    def update(self, df: pd.DataFrame) -> None:
        """Absorb feature rows into the reservoir (call AFTER transform)."""
        if df is None or df.empty:
            return
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
            self._absorb_column(col, vals)

    def bounds(self) -> dict[str, tuple[float, float]]:
        """Current per-column clip bounds derived from PAST data only."""
        out: dict[str, tuple[float, float]] = {}
        for col, arr in self._reservoirs.items():
            if arr.size == 0:
                continue
            out[col] = (
                float(np.quantile(arr, self.limits[0])),
                float(np.quantile(arr, 1.0 - self.limits[1])),
            )
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip df using reservoir quantiles (past-only). No-op until prior data exists."""
        if df is None or df.empty:
            return df
        bounds = self.bounds()
        if not bounds:
            return df.copy()
        out = df.copy()
        for col, (lower, upper) in bounds.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=lower, upper=upper)
        return out

    def serialize(self) -> dict:
        return {
            "limits": list(self.limits),
            "max_samples": self.max_samples,
            "reservoirs": {k: v.tolist() for k, v in self._reservoirs.items()},
            "counts": self._counts,
        }

    @classmethod
    def deserialize(cls, payload: dict) -> "AccumulatingWinsorizer":
        wz = cls(limits=tuple(payload["limits"]), max_samples=payload["max_samples"])
        wz._reservoirs = {k: np.asarray(v, dtype=np.float64) for k, v in payload.get("reservoirs", {}).items()}
        wz._counts = {k: int(v) for k, v in payload.get("counts", {}).items()}
        return wz
