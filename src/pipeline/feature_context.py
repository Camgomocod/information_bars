"""
feature_context.py — Causal rolling context for feature engineering.

The DRB feature set (returns, fractional differentiation, rolling volatility,
ATR, RSI, Bollinger, MACD, volume z-scores, funding rate) all have lookback
windows. When pipelines process month-by-month (or day-by-day), computing
features on an isolated slice loses that history AND makes the first rows of
each slice NaNs that get dropped — a silent data loss.

``FeatureContext`` threads the two pieces of state that make features correct
and causal across period boundaries:

- ``bars_tail``  : last N bars from strictly-prior periods, concatenated ahead
                   of the new bars so rolling windows have real history.
- ``winsorizer`` : an ``AccumulatingWinsorizer`` whose bounds only ever contain
                   data from periods BEFORE the one being transformed.

Usage:
    ctx = FeatureContext(symbol="BTCUSDT")
    feats_jan = ctx.compute_features(jan_bars)
    feats_feb = ctx.compute_features(feb_bars)   # sees Jan in every rolling window
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.features.base_features import BAR_FEATURE_COLUMNS, compute_all_features
from src.normalizers.data_normalizer import AccumulatingWinsorizer
from src.pipeline.logging_setup import get_logger

logger = get_logger("feature_context")

# Enough history to cover: FFD weight window, MACD(12,26,9), RSI(14),
# BB(20), vol(20), ATR(14) plus a warm-up margin.
DEFAULT_MAX_CONTEXT_BARS = 600

FEATURE_COLS = list(BAR_FEATURE_COLUMNS)


@dataclass
class FeatureContext:
    """Causal feature state threaded across chronological periods."""

    symbol: str
    bars_tail: pd.DataFrame = field(default_factory=pd.DataFrame)
    winsorizer: AccumulatingWinsorizer = field(default_factory=lambda: AccumulatingWinsorizer(limits=(0.01, 0.01)))
    max_context_bars: int = DEFAULT_MAX_CONTEXT_BARS

    # -- context maintenance ------------------------------------------------

    def seed_bars(self, prior_bars: pd.DataFrame) -> None:
        """Prime the context with bars STRICTLY BEFORE the next period."""
        if prior_bars is None or prior_bars.empty:
            self.bars_tail = pd.DataFrame()
            return
        cols = [
            c
            for c in ("open_time", "close_time", "open", "high", "low", "close", "n_ticks", "volume", "dollar_value")
            if c in prior_bars.columns
        ]
        frame = prior_bars[cols].copy()
        frame = frame.sort_values("open_time").reset_index(drop=True)
        self.bars_tail = frame.tail(self.max_context_bars).copy()

    def seed_winsorizer(self, prior_features: pd.DataFrame) -> None:
        """Absorb strictly-prior feature rows into the winsorizer bounds."""
        if prior_features is None or prior_features.empty:
            return
        cols = [c for c in FEATURE_COLS if c in prior_features.columns]
        if cols:
            self.winsorizer.update(prior_features[cols])

    def _absorb_bars(self, new_bars: pd.DataFrame) -> None:
        if new_bars is None or new_bars.empty:
            return
        cols = [
            c
            for c in ("open_time", "close_time", "open", "high", "low", "close", "n_ticks", "volume", "dollar_value")
            if c in new_bars.columns
        ]
        combined = pd.concat([self.bars_tail, new_bars[cols]], ignore_index=True)
        combined = combined.sort_values("open_time").reset_index(drop=True)
        self.bars_tail = combined.tail(self.max_context_bars).copy()

    # -- causal feature computation -----------------------------------------

    def compute_features(
        self,
        new_bars: pd.DataFrame,
        positioning_data: dict | None = None,
        positioning_data_dir: str = "data_raw/futures",
        drop_warmup: bool = True,
    ) -> pd.DataFrame:
        """Compute features for ``new_bars`` using ONLY prior history.

        Rolling windows span context + new bars, so the new rows are never
        warm-up-truncated by a period boundary. Only the new bars' rows are
        returned, clipped by winsorizer bounds learned from strictly-prior
        periods. Afterwards the winsorizer and bars tail are updated so the
        NEXT period sees this one as history.
        """
        if new_bars is None or new_bars.empty:
            return pd.DataFrame()

        prior = self.bars_tail.copy()
        prior["_feature_context_is_new"] = False
        current = new_bars.reset_index(drop=True).copy()
        current["_feature_context_is_new"] = True
        combined = pd.concat([prior, current], ignore_index=True)
        combined = combined.sort_values("open_time").reset_index(drop=True)

        kwargs = {"drop_warmup": drop_warmup, "winsorize": False}
        if positioning_data is not None or self.symbol is not None:
            kwargs["symbol"] = self.symbol
            kwargs["positioning_data_dir"] = positioning_data_dir
            if positioning_data is not None:
                kwargs["positioning_data"] = positioning_data

        features = compute_all_features(combined, **kwargs)
        if features is None or features.empty:
            return pd.DataFrame()

        # Clip with strictly-prior bounds only (no-op for the very first period).
        features = self.winsorizer.transform(features)

        if "_feature_context_is_new" in features.columns:
            new_feats = features[features["_feature_context_is_new"]].copy()
            new_feats = new_feats.drop(columns=["_feature_context_is_new"])
        else:
            new_feats = features.tail(len(new_bars)).copy()
        if new_feats.empty:
            if not self.winsorizer.bounds():
                logger.info("symbol=%s first period: no prior winsorizer bounds (no-op clip)", self.symbol)
            return new_feats

        # Absorb this period as history for the NEXT period.
        cols = [c for c in FEATURE_COLS if c in new_feats.columns]
        self.winsorizer.update(new_feats[cols])
        self._absorb_bars(new_bars)

        return new_feats

    def to_state(self) -> dict:
        return {
            "bars_tail": self.bars_tail,
            "winsorizer": self.winsorizer.serialize(),
            "max_context_bars": self.max_context_bars,
        }

    @classmethod
    def from_state(cls, symbol: str, state: dict) -> FeatureContext:
        ctx = cls(
            symbol=symbol,
            bars_tail=state.get("bars_tail", pd.DataFrame()),
            max_context_bars=state.get("max_context_bars", DEFAULT_MAX_CONTEXT_BARS),
        )
        if state.get("winsorizer"):
            ctx.winsorizer = AccumulatingWinsorizer.deserialize(state["winsorizer"])
        return ctx
