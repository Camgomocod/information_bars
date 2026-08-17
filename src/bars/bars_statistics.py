"""
Experimentation System for Dollar Imbalance Bars (DIBs) and Dollar Run Bars (DRBs)
ULTRA-OPTIMIZED VERSION: COMPLETE processing of the month
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Iterator
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import json
import warnings
import gc

warnings.filterwarnings("ignore")

# Style configuration
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class BarsStatistics:
    """Class for computing advanced bar statistics"""

    def compute_returns(self, bars_df: pd.DataFrame) -> pd.Series:
        """Computes log returns of the bars"""
        return np.log(bars_df["close"] / bars_df["close"].shift(1)).dropna()

    def test_normality(self, returns: pd.Series) -> Dict:
        """Normality test"""
        jb_stat, jb_pvalue = stats.jarque_bera(returns)

        if len(returns) < 5000:
            sw_stat, sw_pvalue = stats.shapiro(returns)
        else:
            sw_stat, sw_pvalue = np.nan, np.nan

        return {
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pvalue": float(jb_pvalue),
            "shapiro_wilk_stat": float(sw_stat),
            "shapiro_wilk_pvalue": float(sw_pvalue),
            "skewness": float(stats.skew(returns)),
            "kurtosis": float(stats.kurtosis(returns)),
        }

    def test_autocorrelation(self, returns: pd.Series, lags: int = 20) -> Dict:
        """Autocorrelation test"""
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb_result = acorr_ljungbox(returns, lags=[lags], return_df=True)
        acf_lag1 = returns.autocorr(lag=1)

        return {
            "ljung_box_stat": float(lb_result["lb_stat"].iloc[0]),
            "ljung_box_pvalue": float(lb_result["lb_pvalue"].iloc[0]),
            "acf_lag1": float(acf_lag1),
            "abs_acf_lag1": float(abs(acf_lag1)),
        }

    def test_stationarity(self, returns: pd.Series) -> Dict:
        """Stationarity test"""
        from statsmodels.tsa.stattools import adfuller

        adf_result = adfuller(returns, autolag="AIC")

        return {
            "adf_statistic": float(adf_result[0]),
            "adf_pvalue": float(adf_result[1]),
            "adf_critical_1pct": float(adf_result[4]["1%"]),
            "adf_critical_5pct": float(adf_result[4]["5%"]),
            "is_stationary": bool(adf_result[1] < 0.05),
        }

    def test_variance_homogeneity(
        self, bars_df: pd.DataFrame, n_bins: int = 10
    ) -> Dict:
        """Variance homogeneity test"""
        returns = self.compute_returns(bars_df)

        bin_size = len(returns) // n_bins
        bins = [returns[i * bin_size : (i + 1) * bin_size] for i in range(n_bins)]
        bins = [b for b in bins if len(b) > 0]

        if len(bins) < 2:
            return {
                "levene_stat": np.nan,
                "levene_pvalue": np.nan,
                "variance_ratio": np.nan,
            }

        levene_stat, levene_pvalue = stats.levene(*bins)

        return {
            "levene_stat": float(levene_stat),
            "levene_pvalue": float(levene_pvalue),
            "variance_ratio": float(returns.var() / np.mean([b.var() for b in bins])),
        }

    def compute_sampling_efficiency(self, bars_df: pd.DataFrame) -> Dict:
        """Sampling efficiency metrics"""
        n_ticks = bars_df["n_ticks"].values

        return {
            "mean_ticks": float(np.mean(n_ticks)),
            "std_ticks": float(np.std(n_ticks)),
            "cv_ticks": float(np.std(n_ticks) / np.mean(n_ticks)),
            "min_ticks": int(np.min(n_ticks)),
            "max_ticks": int(np.max(n_ticks)),
            "median_ticks": float(np.median(n_ticks)),
            "iqr_ticks": float(np.percentile(n_ticks, 75) - np.percentile(n_ticks, 25)),
        }

    def compute_bar_metrics(self, bars_df: pd.DataFrame) -> Dict:
        """General bar metrics"""
        returns = self.compute_returns(bars_df)

        return {
            "n_bars": int(len(bars_df)),
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "sharpe_ratio": float(
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0
                else 0
            ),
            "max_drawdown": float(self._compute_max_drawdown(bars_df)),
            "total_volume": float(bars_df["volume"].sum()),
            "total_dollar_value": float(bars_df["dollar_value"].sum()),
        }

    def _compute_max_drawdown(self, bars_df: pd.DataFrame) -> float:
        """Computes the maximum drawdown"""
        cumulative = (1 + self.compute_returns(bars_df)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def compute_all_statistics(self, bars_df: pd.DataFrame) -> Dict:
        """Computes ALL statistics"""
        returns = self.compute_returns(bars_df)

        stats_dict = {
            **self.test_normality(returns),
            **self.test_autocorrelation(returns),
            **self.test_stationarity(returns),
            **self.test_variance_homogeneity(bars_df),
            **self.compute_sampling_efficiency(bars_df),
            **self.compute_bar_metrics(bars_df),
        }

        return stats_dict

    def compute_quality_score(self, stats_dict: Dict) -> float:
        """
        DEPRECATED — p-value based; kept for backward compatibility (compare_bar_types).

        Prefer compute_quality_score_fast(), which is effect-size based and does
        not reward "not rejecting" null hypotheses.
        """
        score = 0

        acf = abs(stats_dict.get("acf_lag1", 1))
        score += max(0, 30 * (1 - acf))

        if stats_dict.get("is_stationary", False):
            score += 25

        levene_p = stats_dict.get("levene_pvalue", 0)
        score += min(20, levene_p * 20)

        cv = stats_dict.get("cv_ticks", 2)
        score += max(0, 15 * (1 - min(cv, 1)))

        jb_p = stats_dict.get("jarque_bera_pvalue", 0)
        score += min(10, jb_p * 10)

        return score

    def compute_quality_score_fast(self, bars_df: pd.DataFrame) -> float:
        """
        Calculates quality score directly from bars_df without passing
        through compute_all_statistics().

        v2 (effect-size based): rewards the MAGNITUDE of the underlying effects
        rather than "not rejecting" a null hypothesis. p-values alone are poor
        rewards: they depend on sample size and conflate weak evidence with a
        good bar. Components:

          1. Autocorrelation of returns (20 pts)     — against N(0, 1/sqrt(n)) band.
          2. Autocorrelation of squared returns (20) — captures non-linear
             dependence / volatility clustering (also heteroscedasticity).
          3. Variance homogeneity (20 pts)           — inter-bin variance ratio.
          4. Stationarity (10 pts)                   — standardized ADF magnitude.
          5. Sampling efficiency (10 pts)            — CV of ticks.
          6. Non-normality magnitude (10 pts)        — |skew| + |excess kurtosis|.

        The efficiency terms dominate so that autocorrelated or heteroscedastic
        series score clearly lower even when stationary and normally distributed.
        Range ≈ 0-100.
        """
        returns = self.compute_returns(bars_df)
        n = len(returns)
        score = 0.0

        def _acf_points(series: pd.Series, pts: float) -> float:
            """Score pts scaled by how far |acf(lag=1)| is above the noise band."""
            if n < 4:
                return 0.0
            acf = abs(float(series.autocorr(lag=1)))
            se = 1.0 / np.sqrt(n)
            noise_band = 1.96 * se
            if acf <= noise_band:
                return pts
            excess = (acf - noise_band) / max(noise_band, 1e-9)
            return max(0.0, pts * (1.0 - excess))

        # 1+2. Autocorrelation of returns and squared returns (40 pts total)
        score += _acf_points(returns, 20.0)
        score += _acf_points(returns ** 2, 20.0)

        # 3. Variance homogeneity (20 pts) — inter-bin variance ratio near 1
        n_bins = 10
        bin_size = n // n_bins
        if bin_size > 0:
            bins = [returns.iloc[i * bin_size : (i + 1) * bin_size] for i in range(n_bins)]
            bins = [b for b in bins if len(b) > 0]
            if len(bins) >= 2:
                var_ratios = [np.var(b.values) for b in bins]
                var_max, var_min = max(var_ratios), min(var_ratios)
                if var_min > 0:
                    ratio = var_max / var_min
                    # ratio == 1 → full 20 pts; ratio >= 10 → 0 pts (log scale)
                    score += max(0.0, 20.0 * (1.0 - np.log10(ratio) / 1.0))

        # 4. Stationarity (10 pts) — standardized ADF statistic
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(returns, maxlag=5, autolag=None)
        adf_stat = float(adf_result[0])
        crit_1pct = float(adf_result[4]["1%"])
        if adf_stat <= crit_1pct:
            # Fully stationary: full points.
            score += 10.0
        else:
            # Penalize proportionally as adf rises from crit toward 0.
            # NOTE: crit_1pct is negative — never use max(crit_1pct, eps) as
            # denominator, it collapses to eps and the ratio explodes by 1e9.
            denom = -crit_1pct if crit_1pct < 0 else 1.0
            penalty = min(1.0, max(0.0, (adf_stat - crit_1pct) / denom))
            score += 10.0 * (1.0 - penalty)

        # 5. Sampling efficiency (10 pts) — CV of ticks
        n_ticks = bars_df["n_ticks"].values
        mean_ticks = np.mean(n_ticks)
        cv = np.std(n_ticks) / mean_ticks if mean_ticks > 0 else 2.0
        score += max(0.0, 10.0 * (1.0 - min(cv, 1.0)))

        # 6. Non-normality magnitude (10 pts) — |skew| + |excess kurtosis|
        skew = abs(float(stats.skew(returns)))
        kurt = abs(float(stats.kurtosis(returns)))
        combined = skew + kurt
        score += max(0.0, 10.0 * (1.0 - combined / 5.0))

        return score
