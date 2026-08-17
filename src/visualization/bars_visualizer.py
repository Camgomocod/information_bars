import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import pandas as pd

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class BarsVisualizer:
    """Class responsible for all bar visualization"""

    def __init__(self, save_path=None):
        self.save_path = save_path

    def plot_comparison(self, df: pd.DataFrame, bars_dict: dict):
        """
        Compares multiple types of bars

        Args:
            df: DataFrame with original trades
            bars_dict: Dict with format {'name': (bars_df, color, marker)}
                      Example: {'TIBs': (tibs_df, 'red', 'x')}
        """
        n_bars = len(bars_dict)
        fig, axes = plt.subplots(n_bars, 1, figsize=(14, 3.5 * n_bars), sharex=True)

        # If there is only one type of bar, axes is not a list
        if n_bars == 1:
            axes = [axes]

        for idx, (bar_name, (bars_df, color, marker)) in enumerate(bars_dict.items()):
            axes[idx].plot(
                df["timestamp"],
                df["price"],
                alpha=0.3,
                label="Trades",
                linewidth=0.5,
                color="gray",
            )
            axes[idx].scatter(
                bars_df["close_time"],
                bars_df["close"],
                color=color,
                s=30,
                label=f"{bar_name} (n={len(bars_df)})",
                alpha=0.7,
                marker=marker,
            )
            axes[idx].set_ylabel("Price (USD)")
            axes[idx].set_title(bar_name)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time")
        plt.tight_layout()

        if self.save_path:
            plt.savefig(
                f"{self.save_path}/comparison.png", dpi=300, bbox_inches="tight"
            )
        plt.show()

    def plot_single_bar(
        self, df: pd.DataFrame, bars_df: pd.DataFrame, bar_type: str, color="red"
    ):
        """Plot for a specific type of bar"""
        plt.figure(figsize=(14, 6))
        plt.plot(
            df["timestamp"],
            df["price"],
            alpha=0.4,
            label="Trades",
            linewidth=0.8,
            color="gray",
        )
        plt.scatter(
            bars_df["close_time"],
            bars_df["close"],
            color=color,
            s=40,
            label=f"{bar_type} Close (n={len(bars_df)})",
            zorder=5,
        )
        plt.legend()
        plt.title(f"{bar_type} - BTCUSDT")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if self.save_path:
            plt.savefig(
                f"{self.save_path}/{bar_type}.png", dpi=300, bbox_inches="tight"
            )
        plt.show()

    def candle_chart(self, bars_df: pd.DataFrame, bar_type: str):
        """Plots candles for the bars"""
        df_plot = bars_df.copy()

        # If close_time is not the index, set it
        if "close_time" in df_plot.columns:
            df_plot.set_index("close_time", inplace=True)

        # Ensure the index is datetime
        if not isinstance(df_plot.index, pd.DatetimeIndex):
            df_plot.index = pd.to_datetime(df_plot.index)

        # Verify we have the necessary columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df_plot.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        mpf.plot(
            df_plot[required_cols],
            type="candle",
            style="charles",
            volume=True,
            title=f"{bar_type} - BTCUSDT",
            ylabel="Price (USD)",
            ylabel_lower="Volume",
            savefig=f"{self.save_path}/{bar_type}_candles.png"
            if self.save_path
            else None,
        )

    def plot_bar_statistics(self, bars_df: pd.DataFrame, bar_type: str):
        """Statistics of the generated bars"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # Distribution of ticks per bar
        axes[0, 0].hist(bars_df["n_ticks"], bins=50, color="steelblue", alpha=0.7)
        axes[0, 0].set_title("Distribution of Ticks per Bar")
        axes[0, 0].set_xlabel("Number of Ticks")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True, alpha=0.3)

        # Distribution of volume per bar
        axes[0, 1].hist(bars_df["volume"], bins=50, color="green", alpha=0.7)
        axes[0, 1].set_title("Distribution of Volume per Bar")
        axes[0, 1].set_xlabel("Volume")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)

        # Distribution of dollar value per bar
        axes[1, 0].hist(bars_df["dollar_value"], bins=50, color="orange", alpha=0.7)
        axes[1, 0].set_title("Distribution of Dollar Value per Bar")
        axes[1, 0].set_xlabel("Dollar Value")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True, alpha=0.3)

        # Time series of the number of ticks
        axes[1, 1].plot(
            bars_df["close_time"],
            bars_df["n_ticks"],
            color="purple",
            linewidth=1,
            alpha=0.7,
        )
        axes[1, 1].set_title("Ticks per Bar Over Time")
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Number of Ticks")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f"{bar_type} - Statistics", fontsize=16, y=1.00)
        plt.tight_layout()

        if self.save_path:
            plt.savefig(
                f"{self.save_path}/{bar_type}_stats.png", dpi=300, bbox_inches="tight"
            )
        plt.show()
