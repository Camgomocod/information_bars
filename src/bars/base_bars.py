import numpy as np
import pandas as pd


class BaseBars:
    MIN_TICKS = None  # loaded lazily from config

    @classmethod
    def _get_min_ticks(cls) -> int:
        if cls.MIN_TICKS is None:
            try:
                from src.config import get_bar_config
                min_ticks = get_bar_config("drb").get("min_ticks", 400)
                cls.MIN_TICKS = int(min_ticks) if min_ticks is not None else 400
            except Exception:
                cls.MIN_TICKS = 400
        return cls.MIN_TICKS

    def __init__(self):
        pass

    def get_prob_buy(self, bar_b_t, current_prob, lambada_):
        recent_prob = float((bar_b_t == 1).mean())
        return (1 - lambada_**2) * recent_prob + lambada_**2 * current_prob

    @staticmethod
    def get_tick_directions(
        df: pd.DataFrame, direction_mode: str = "tick_rule"
    ) -> pd.Series:
        """Calculates the direction of each tick (buy=+1, sell=-1).

        Parameters
        ----------
        direction_mode : str
            - ``is_buyer_maker`` : use Binance ``is_buyer_maker``. A maker buyer
              means the taker SOLD, so ``is_buyer_maker=True`` → sell (-1) and
              ``is_buyer_maker=False`` → buy (+1). Falls back to ``tick_rule``
              when the column is absent.
            - ``tick_rule`` : infer direction from the price sign convention
              (sign of price diff, zero-diffs forward-filled, first tick = buy).

        The aggressive-taker convention is what the run-bar theory expects:
        b_t = +1 when a buyer crosses the spread (market buy), -1 otherwise.
        """
        if direction_mode == "is_buyer_maker" and "is_buyer_maker" in df.columns:
            maker = df["is_buyer_maker"].astype(bool)
            return pd.Series(
                np.where(maker, -1, 1),
                index=df.index,
                dtype=np.int8,
                name="b_t",
            )

        price_diff = df["price"].diff()
        b_t = np.sign(price_diff)
        b_t = b_t.replace(0, np.nan).ffill().fillna(1)
        return b_t.astype(np.int8)

    def _form_bar(self, bar):
        """Forms an OHLC bar from a set of ticks"""
        return {
            "open_time": bar["timestamp"].iloc[0],
            "close_time": bar["timestamp"].iloc[-1],
            "open": bar["price"].iloc[0],
            "high": bar["price"].max(),
            "low": bar["price"].min(),
            "close": bar["price"].iloc[-1],
            "n_ticks": len(bar),
            "volume": bar["quantity"].sum(),
            "dollar_value": bar["dollar_value"].sum(),
        }
