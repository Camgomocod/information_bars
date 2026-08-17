"""
compare_bar_types.py — Statistical comparison of DRBs vs Time Bars

Generates a PDF report with side-by-side statistics and visualizations
for Dollar Run Bars and Time Bars (1m/5m/15m/1h).

Usage:
    micromamba run -n trading-core python scripts/compare_bar_types.py \
        --symbol BTCUSDT --start 2020-01-01 --end 2020-01-31

    micromamba run -n trading-core python scripts/compare_bar_types.py \
        --symbol BTCUSDT --start 2020-01-01 --end 2020-02-29 \
        --intervals 1 5 15 60 --output-dir reports
"""

import sys
import argparse
import gc
import warnings
from pathlib import Path
from datetime import date, datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.bars.bars_statistics import BarsStatistics

plt = None


def _import_plt():
    global plt
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_pdf as pdfpages
    return plt


def load_bars(data_dir: Path, symbol: str, pattern: str,
              start: date, end: date) -> pd.DataFrame | None:
    data_dir = data_dir / symbol
    if not data_dir.exists():
        return None
    files = sorted(data_dir.glob(pattern))
    files = [f for f in files if _date_from_filename(f, start, end)]
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    ts_col = "open_time" if "open_time" in df.columns else "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)
    return df


def _date_from_filename(path: Path, start: date, end: date) -> bool:
    parts = path.stem.split("_")
    for p in parts:
        try:
            d = datetime.strptime(p, "%Y-%m-%d").date()
            return start <= d <= end
        except ValueError:
            continue
    return False


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "timestamp" in df.columns and "open_time" not in df.columns:
        rename["timestamp"] = "open_time"
    if rename:
        df = df.rename(columns=rename)
    for col in ["n_ticks", "volume", "dollar_value"]:
        if col not in df.columns:
            df[col] = 0
    return df


def compute_comparison(bars_dict: dict[str, pd.DataFrame],
                       stats_calc: BarsStatistics) -> pd.DataFrame:
    rows = []
    for label, df in bars_dict.items():
        if df is None or df.empty:
            continue
        df = normalize_columns(df)
        s = stats_calc.compute_all_statistics(df)
        qs = stats_calc.compute_quality_score(s)
        s["quality_score"] = qs
        s["bar_type"] = label
        rows.append(s)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("bar_type")


def generate_pdf_report(
    comp_df: pd.DataFrame,
    bars_dict: dict[str, pd.DataFrame],
    symbol: str,
    start: date,
    end: date,
    output_path: Path,
    stats_calc: BarsStatistics = None,
):
    if stats_calc is None:
        stats_calc = BarsStatistics()
    pl = _import_plt()
    from matplotlib.backends.backend_pdf import PdfPages

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
    bar_types = [bt for bt in bars_dict if bars_dict[bt] is not None and not bars_dict[bt].empty]

    with PdfPages(output_path) as pdf:

        # ── Page 1: Cover ──
        fig, ax = pl.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.85, "Bars Comparison Report", fontsize=28, ha="center", fontweight="bold")
        ax.text(0.5, 0.72, f"{symbol}", fontsize=20, ha="center", color="#555")
        ax.text(0.5, 0.62, f"{start}  →  {end}", fontsize=16, ha="center", color="#777")
        ax.text(0.5, 0.50, " vs ".join(bar_types), fontsize=14, ha="center",
                color="#3498db", style="italic")
        ax.text(0.5, 0.35, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                fontsize=10, ha="center", color="#aaa")
        ax.text(0.5, 0.12,
                "Dollar Run Bars (DRB) sample markets based on volume flow.\n"
                "Time Bars sample at fixed clock intervals (1m, 5m, 15m, 1h).\n"
                "This report compares statistical properties across bar types.",
                fontsize=10, ha="center", color="#888", linespacing=1.5)
        pdf.savefig(fig)
        pl.close(fig)

        # ── Page 2: Statistics Table ──
        fig, ax = pl.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.96, "Comparative Statistics", fontsize=18, ha="center",
                fontweight="bold")

        metrics = [
            ("n_bars", "N Bars", "{:,.0f}"),
            ("mean_return", "Mean Return", "{:.6f}"),
            ("std_return", "Std Return", "{:.4f}"),
            ("sharpe_ratio", "Sharpe Ratio (252d)", "{:.2f}"),
            ("max_drawdown", "Max Drawdown", "{:.2%}"),
            ("skewness", "Skewness", "{:.4f}"),
            ("kurtosis", "Excess Kurtosis", "{:.4f}"),
            ("jarque_bera_stat", "Jarque-Bera Stat", "{:.2f}"),
            ("jarque_bera_pvalue", "Jarque-Bera p-value", "{:.4f}"),
            ("ljung_box_stat", "Ljung-Box Stat (lag 20)", "{:.2f}"),
            ("ljung_box_pvalue", "Ljung-Box p-value", "{:.4f}"),
            ("acf_lag1", "ACF(1)", "{:.4f}"),
            ("adf_statistic", "ADF Statistic", "{:.4f}"),
            ("adf_pvalue", "ADF p-value", "{:.6f}"),
            ("is_stationary", "Is Stationary", "{}"),
            ("levene_stat", "Levene Stat", "{:.4f}"),
            ("levene_pvalue", "Levene p-value", "{:.4f}"),
            ("mean_ticks", "Mean Ticks/Bar", "{:,.0f}"),
            ("cv_ticks", "CV Ticks", "{:.4f}"),
            ("quality_score", "Quality Score", "{:.2f}"),
        ]

        col_labels = ["Metric"] + bar_types
        col_widths = [0.30] + [0.70 / len(bar_types)] * len(bar_types)

        table_data = []
        for key, label, fmt in metrics:
            row = [label]
            for bt in bar_types:
                val = comp_df.loc[bt, key] if bt in comp_df.index and key in comp_df.columns else "N/A"
                if isinstance(val, bool):
                    row.append("✓" if val else "✗")
                elif isinstance(val, (int, float)) and not np.isnan(val):
                    row.append(fmt.format(val))
                else:
                    row.append("N/A")
            table_data.append(row)

        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="upper center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 0.55)

        for (i, j), cell in tbl.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight="bold", fontsize=9)
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white")
            elif j == 0:
                cell.set_text_props(fontweight="bold", fontsize=8)
                cell.set_facecolor("#ecf0f1")
            elif j > 0 and i > 0:
                val = table_data[i - 1][j]
                metric_key = metrics[i - 1][0] if i - 1 < len(metrics) else ""
                better_when_higher = metric_key in ("quality_score", "is_stationary", "sharpe_ratio")
                if val not in ("N/A",):
                    try:
                        vals = [table_data[i - 1][k] for k in range(1, len(bar_types) + 1)]
                        nums = [float(v) for v in vals if v not in ("N/A", "✗", "✓")]
                        if nums:
                            best = max(nums) if better_when_higher else min(nums)
                            if abs(float(val) - best) < 1e-6:
                                cell.set_facecolor("#d5f5e3")
                    except (ValueError, TypeError):
                        pass

        pdf.savefig(fig)
        pl.close(fig)

        # ── Page 3: Return Distribution ──
        fig, axes = pl.subplots(1, 2, figsize=(11, 4.5))
        for ax, (bt, df) in zip(axes, bars_dict.items()):
            if df is None or df.empty:
                ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
                continue
            df = normalize_columns(df)
            ret = stats_calc.compute_returns(df)
            ax.hist(ret, bins=80, density=True, alpha=0.65, color=colors[0], edgecolor="white")
            mu, sigma = ret.mean(), ret.std()
            x = np.linspace(ret.min(), ret.max(), 200)
            ax.plot(x, (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                    color="red", lw=1.5, label="Normal fit")
            ax.set_title(f"{bt} — Returns Distribution", fontsize=12, fontweight="bold")
            ax.set_xlabel("Log Return")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            s = comp_df.loc[bt] if bt in comp_df.index else None
            if s is not None:
                ax.text(0.97, 0.93, f"Skew={s['skewness']:.3f}\nKurt={s['kurtosis']:.3f}",
                        transform=ax.transAxes, fontsize=8, ha="right", va="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        pl.tight_layout()
        pdf.savefig(fig)
        pl.close(fig)

        # ── Page 4: QQ-Plot ──
        n_bt = len(bar_types)
        fig, axes = pl.subplots(2, max(2, (n_bt + 1) // 2), figsize=(11, 8))
        axes = axes.flatten()
        for i, bt in enumerate(bar_types):
            df = bars_dict[bt]
            if df is None or df.empty:
                axes[i].text(0.5, 0.5, "No data", ha="center", transform=axes[i].transAxes)
                continue
            df = normalize_columns(df)
            ret = stats_calc.compute_returns(df)
            from scipy import stats as scistats
            scistats.probplot(ret, dist="norm", plot=axes[i])
            axes[i].set_title(f"{bt}", fontsize=10)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        pl.suptitle("Q-Q Plots (Normality)", fontsize=14, fontweight="bold", y=1.01)
        pl.tight_layout()
        pdf.savefig(fig)
        pl.close(fig)

        # ── Page 5: ACF ──
        fig, axes = pl.subplots(2, max(2, (n_bt + 1) // 2), figsize=(11, 8))
        axes = axes.flatten()
        for i, bt in enumerate(bar_types):
            df = bars_dict[bt]
            if df is None or df.empty:
                axes[i].text(0.5, 0.5, "No data", ha="center", transform=axes[i].transAxes)
                continue
            df = normalize_columns(df)
            ret = stats_calc.compute_returns(df)
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(ret, lags=40, ax=axes[i], alpha=0.05)
            axes[i].set_title(f"{bt}", fontsize=10)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        pl.suptitle("Autocorrelation Function (ACF)", fontsize=14, fontweight="bold", y=1.01)
        pl.tight_layout()
        pdf.savefig(fig)
        pl.close(fig)

        # ── Page 6: Equity Curves ──
        fig, ax = pl.subplots(figsize=(11, 5))
        for i, bt in enumerate(bar_types):
            df = bars_dict[bt]
            if df is None or df.empty:
                continue
            df = normalize_columns(df)
            ret = stats_calc.compute_returns(df)
            eq = (1 + ret).cumprod()
            ax.plot(eq.index, eq.values, label=bt, color=colors[i % len(colors)], lw=0.8)
        ax.set_title("Cumulative Returns (Equity Curves)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Bar Index")
        ax.set_ylabel("Cumulative Return")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        pl.tight_layout()
        pdf.savefig(fig)
        pl.close(fig)

        # ── Page 7: Drawdown ──
        fig, ax = pl.subplots(figsize=(11, 5))
        for i, bt in enumerate(bar_types):
            df = bars_dict[bt]
            if df is None or df.empty:
                continue
            df = normalize_columns(df)
            ret = stats_calc.compute_returns(df)
            cum = (1 + ret).cumprod()
            running_max = cum.expanding().max()
            dd = (cum - running_max) / running_max
            ax.plot(dd.index, dd.values, label=bt, color=colors[i % len(colors)], lw=0.8)
        ax.set_title("Drawdown Curves", fontsize=13, fontweight="bold")
        ax.set_xlabel("Bar Index")
        ax.set_ylabel("Drawdown")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        pl.tight_layout()
        pdf.savefig(fig)
        pl.close(fig)

        # ── Page 8: Bars per Day ──
        fig, ax = pl.subplots(figsize=(11, 5))
        bp_data = []
        bp_labels = []
        for bt in bar_types:
            df = bars_dict[bt]
            if df is None or df.empty:
                continue
            df = normalize_columns(df)
            ts_col = "open_time"
            daily_counts = df[ts_col].dt.date.value_counts().sort_index()
            bp_data.append(daily_counts.values)
            bp_labels.append(bt)
        if bp_data:
            bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_title("Bars per Day", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Bars")
            ax.grid(True, axis="y", alpha=0.3)
        pl.tight_layout()
        pdf.savefig(fig)
        pl.close(fig)

        # ── Page 9: Quality Score Breakdown ──
        fig, ax = pl.subplots(figsize=(11, 5))
        qs_metrics = ["quality_score"]
        x = np.arange(len(bar_types))
        width = 0.35
        scores = []
        for bt in bar_types:
            if bt in comp_df.index:
                scores.append(comp_df.loc[bt, "quality_score"])
            else:
                scores.append(0)
        bars = ax.bar(x, scores, width, color=colors[:len(bar_types)], alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_types, fontsize=10)
        ax.set_title("Quality Score Comparison", fontsize=13, fontweight="bold")
        ax.set_ylabel("Quality Score (0–100)")
        ax.set_ylim(0, 105)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{score:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        pl.tight_layout()
        pdf.savefig(fig)
        pl.close(fig)

        # ── Page 10: Tick Efficiency (CV Ticks) ──
        fig, ax = pl.subplots(figsize=(11, 5))
        cv_data = []
        cv_labels = []
        for bt in bar_types:
            df = bars_dict[bt]
            if df is None or df.empty:
                continue
            if "n_ticks" not in df.columns:
                cv_data.append([0])
                cv_labels.append(bt)
                continue
            cv = df["n_ticks"]
            cv_data.append(cv.values)
            cv_labels.append(bt)
        if cv_data:
            bp = ax.boxplot(cv_data, labels=cv_labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_title("Ticks per Bar Distribution", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Ticks")
            ax.grid(True, axis="y", alpha=0.3)
        pl.tight_layout()
        pdf.savefig(fig)
        pl.close(fig)

    print(f"   📄 Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DRBs vs Time Bars statistically"
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2020-01-31")
    parser.add_argument("--drb-dir", default="data/bars/dollar_run")
    parser.add_argument("--time-dir", default="data/bars/time")
    parser.add_argument("--intervals", nargs="+", type=int, default=[1, 60])
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_calc = BarsStatistics()

    print("\n" + "=" * 60)
    print("📊 BAR TYPE COMPARISON")
    print("=" * 60)
    print(f"Symbol   : {args.symbol}")
    print(f"Period   : {start} → {end}")
    print(f"Intervals: {', '.join(f'{i}min' for i in args.intervals)}")
    print(f"DRB dir  : {args.drb_dir}")
    print(f"Time dir : {args.time_dir}")
    print("=" * 60)

    bars_dict = {}

    # Load DRBs
    drb_dir = Path(args.drb_dir)
    drb_data = load_bars(drb_dir, args.symbol, f"{args.symbol}_*_drbs.parquet", start, end)
    if drb_data is not None and not drb_data.empty:
        bars_dict["DRB"] = drb_data
        print(f"   ✅ DRB: {len(drb_data):,} bars loaded")
    else:
        print(f"   ⚠️  DRB: no data found")

    # Load Time Bars
    time_dir = Path(args.time_dir)
    for interval in args.intervals:
        tb_data = load_bars(
            time_dir, args.symbol,
            f"{args.symbol}_*_time_{interval}min.parquet", start, end
        )
        if tb_data is not None and not tb_data.empty:
            bars_dict[f"T{interval}"] = tb_data
            print(f"   ✅ T{interval}: {len(tb_data):,} bars loaded")
        else:
            print(f"   ⚠️  T{interval}: no data found")

    if len(bars_dict) < 2:
        print("\n❌ Need at least 2 bar types with data to compare")
        sys.exit(1)

    if args.dry_run:
        print("\n🔷 DRY-RUN — statistics computed, no PDF saved")
        comp_df = compute_comparison(bars_dict, stats_calc)
        print(comp_df.to_string())
        return

    # Compute statistics
    comp_df = compute_comparison(bars_dict, stats_calc)
    print(f"\n   📊 Statistics computed for {len(comp_df)} bar types")

    # Save CSV
    csv_path = output_dir / f"comparison_{args.symbol}_{start}_{end}.csv"
    comp_df.to_csv(csv_path)
    print(f"   💾 CSV saved: {csv_path}")

    # Generate PDF
    pdf_path = output_dir / f"comparison_{args.symbol}_{start}_{end}.pdf"
    generate_pdf_report(comp_df, bars_dict, args.symbol, start, end, pdf_path, stats_calc)

    # Print text summary
    print("\n" + "=" * 60)
    print("📋 QUICK SUMMARY")
    print("=" * 60)
    for bt in comp_df.index:
        row = comp_df.loc[bt]
        print(f"  {bt:<8} | Bars:{int(row['n_bars']):>6,} | "
              f"Sharpe:{row['sharpe_ratio']:>6.2f} | "
              f"ACF(1):{row['acf_lag1']:>7.4f} | "
              f"Stationary:{'✓' if row['is_stationary'] else '✗'} | "
              f"QS:{row['quality_score']:>5.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
