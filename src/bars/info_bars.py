import numpy as np
import pandas as pd
from numba import jit

from src.bars.bar_audit import BarAudit, check_bar_invariants
from src.bars.base_bars import BaseBars as bb


@jit(nopython=True)
def _compute_dibs_core(b_t, dollar_values, exp_lambda, init_exp_T, min_ticks):
    """
    Numba-optimized core for Dollar Imbalance Bars (DIBs)

    López de Prado Theory:
    θ_T = Σ(b_t * v_t) where v_t is dollar_value
    E[θ_T] = E[T] * E[v_t] * (2*P[b_t=1] - 1)

    Implemented corrections:
    1. EWMA with correct weight: new * (1-λ) + old * λ
    2. E[v_t] calculated over ALL ticks (not just buys)
    3. Threshold without artificial normalization
    """
    n = len(b_t)

    # Inicialización con warm-up
    warm_up = min(100, n // 10)
    E_T = init_exp_T

    # Calcular E[v_t] inicial sobre todos los ticks
    E_v = np.mean(dollar_values[:warm_up])

    # Calcular P[b_t=1] inicial
    P_buy = np.sum(b_t[:warm_up] == 1) / warm_up

    # Arrays para almacenar resultados
    bar_indices = []
    theta = 0.0
    last_index = 0

    for i in range(1, n):
        # Acumular imbalance ponderado por dollar value
        theta += b_t[i] * dollar_values[i]

        # Calcular threshold según López de Prado
        # E[θ_T] = E[T] * E[v_t] * |2*P[b_t=1] - 1|
        expected_imbalance = E_T * E_v * abs(2 * P_buy - 1)

        # Condición de parada
        ticks_in_bar = i - last_index + 1
        if abs(theta) >= expected_imbalance and ticks_in_bar >= min_ticks:
            bar_indices.append((last_index, i + 1))

            # Actualizar expectativas con EWMA
            T_actual = ticks_in_bar
            E_T = (1 - exp_lambda) * T_actual + exp_lambda * E_T

            # E[v_t] sobre TODOS los ticks de la barra
            E_v_new = np.mean(dollar_values[last_index : i + 1])
            E_v = (1 - exp_lambda) * E_v_new + exp_lambda * E_v

            # P[b_t=1] sobre la barra actual
            buy_count = np.sum(b_t[last_index : i + 1] == 1)
            P_buy_new = buy_count / T_actual
            P_buy = (1 - exp_lambda) * P_buy_new + exp_lambda * P_buy

            # Reset
            theta = 0.0
            last_index = i + 1

    return bar_indices


@jit(nopython=True)
def _compute_drbs_core(b_t, dollar_values, exp_lambda, init_exp_T, min_ticks):
    """
    Numba-optimized core for Dollar Run Bars (DRBs)

    López de Prado Theory:
    θ_T = max{Σ(d_t|b_t=1), |Σ(d_t|b_t=-1)|}
    E[θ_T] = E[T] * max{P[b_t=1]*E[d_t|b_t=1], P[b_t=-1]*E[d_t|b_t=-1]}

    Implemented corrections:
    1. Runs are NOT offset (separate accumulation)
    2. E[d_t|b_t=1] calculated ONLY over buy ticks
    3. E[d_t|b_t=-1] calculated ONLY over sell ticks
    """
    n = len(b_t)

    # Inicialización con warm-up
    warm_up = min(100, n // 10)
    E_T = init_exp_T

    # Calcular estadísticas iniciales separadas por tipo de tick
    buy_mask = b_t[:warm_up] == 1
    sell_mask = b_t[:warm_up] == -1

    buy_count = np.sum(buy_mask)
    sell_count = np.sum(sell_mask)

    P_buy = buy_count / warm_up

    # E[d_t|b_t=1] y E[d_t|b_t=-1]
    if buy_count > 0:
        E_d_buy = np.mean(dollar_values[:warm_up][buy_mask])
    else:
        E_d_buy = np.mean(dollar_values[:warm_up])

    if sell_count > 0:
        E_d_sell = np.mean(dollar_values[:warm_up][sell_mask])
    else:
        E_d_sell = np.mean(dollar_values[:warm_up])

    # Arrays para almacenar resultados
    bar_indices = []
    buy_run = 0.0  # Σ(d_t|b_t=1)
    sell_run = 0.0  # Σ(d_t|b_t=-1)
    last_index = 0

    for i in range(1, n):
        d_t = dollar_values[i]

        # Acumular runs SIN offsetting
        if b_t[i] == 1:
            buy_run += d_t
        else:  # b_t[i] == -1
            sell_run += d_t

        # θ_T = max{buy_run, sell_run}
        theta = max(buy_run, sell_run)

        # E[θ_T] = E[T] * max{P[b_t=1]*E[d_t|b_t=1], P[b_t=-1]*E[d_t|b_t=-1]}
        expected_buy = P_buy * E_d_buy
        expected_sell = (1 - P_buy) * E_d_sell
        expected_theta = E_T * max(expected_buy, expected_sell)

        # Condición de parada
        ticks_in_bar = i - last_index + 1
        if theta >= expected_theta and ticks_in_bar >= min_ticks:
            bar_indices.append((last_index, i + 1))

            # Actualizar E[T]
            T_actual = ticks_in_bar
            E_T = (1 - exp_lambda) * T_actual + exp_lambda * E_T

            # Actualizar probabilidades y valores esperados
            bar_b_t = b_t[last_index : i + 1]
            bar_dollars = dollar_values[last_index : i + 1]

            buy_mask_bar = bar_b_t == 1
            sell_mask_bar = bar_b_t == -1

            buy_count_bar = np.sum(buy_mask_bar)
            sell_count_bar = np.sum(sell_mask_bar)

            P_buy_new = buy_count_bar / T_actual
            P_buy = (1 - exp_lambda) * P_buy_new + exp_lambda * P_buy

            if buy_count_bar > 0:
                E_d_buy_new = np.mean(bar_dollars[buy_mask_bar])
                E_d_buy = (1 - exp_lambda) * E_d_buy_new + exp_lambda * E_d_buy

            if sell_count_bar > 0:
                E_d_sell_new = np.mean(bar_dollars[sell_mask_bar])
                E_d_sell = (1 - exp_lambda) * E_d_sell_new + exp_lambda * E_d_sell

            # Reset
            buy_run = 0.0
            sell_run = 0.0
            last_index = i + 1

    return bar_indices


@jit(nopython=True)
def _compute_drbs_core_audited(b_t, dollar_values, exp_lambda, init_exp_T, min_ticks):
    """
    Numba DRB core that also returns the number of residual ticks.

    Residual = ticks after the last closed bar. Reported (never silently
    dropped) so callers can reconcile clean_ticks == consumed + residual.
    """
    bar_indices = _compute_drbs_core(b_t, dollar_values, exp_lambda, init_exp_T, min_ticks)
    residual = len(b_t)
    if len(bar_indices) > 0:
        residual = len(b_t) - bar_indices[-1][1]
    return bar_indices, residual


class OptimizedInfoRunBars:
    """
    Numba-optimized implementation of Dollar Imbalance Bars (DIBs)
    and Dollar Run Bars (DRBs) according to López de Prado's theory.

    Main corrections:
    1. EWMA with correct order: (1-λ)*new + λ*old
    2. DIBs: E[v_t] over all ticks, not just buys
    3. DRBs: Separate runs without offsetting
    4. Threshold without artificial normalizations
    """

    def __init__(self, save_path):
        self.save_path = save_path
        self.base_bars = bb()

    def get_dibs(
        self, df: pd.DataFrame, exp_lambda=0.9, init_exp_T=1600,
        direction_mode: str = "tick_rule",
    ) -> pd.DataFrame:
        """
        Dollar Imbalance Bars (DIBs)

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns: price, quantity, timestamp
        exp_lambda : float
            EWMA decay factor (0.9 = 90% weight to the past)
        init_exp_T : int
            Initial expected value of ticks per bar
        direction_mode : str
            "tick_rule" (price sign) or "is_buyer_maker" (Binance flag).

        Returns:
        --------
        pd.DataFrame with OHLCV bars
        """
        df = df.copy()
        df["b_t"] = self.base_bars.get_tick_directions(df, direction_mode)

        # Asegurar columna dollar_value
        if "dollar_value" not in df.columns:
            # Calcular on-the-fly para ahorrar espacio en raw data
            df["dollar_value"] = df["price"] * df["quantity"]

        # Convertir a arrays numpy para Numba
        b_t = df["b_t"].values
        dollar_values = df["dollar_value"].values

        # Computar barras con Numba
        bar_indices = _compute_dibs_core(
            b_t, dollar_values, exp_lambda, init_exp_T, bb._get_min_ticks()
        )

        # Formar barras
        bars = []
        for start, end in bar_indices:
            bar = df.iloc[start:end]
            bars.append(self.base_bars._form_bar(bar))

        return pd.DataFrame(bars)

    def get_drbs(
        self, df: pd.DataFrame, exp_lambda=0.9, init_exp_T=1600,
        direction_mode: str = "tick_rule",
    ) -> pd.DataFrame:
        """
        Dollar Run Bars (DRBs)

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns: price, quantity, timestamp
        exp_lambda : float
            EWMA decay factor (0.9 = 90% weight to the past)
        init_exp_T : int
            Initial expected value of ticks per bar
        direction_mode : str
            "tick_rule" (price sign) or "is_buyer_maker" (Binance flag).

        Returns:
        --------
        pd.DataFrame with OHLCV bars
        """
        df = df.copy()
        df["b_t"] = self.base_bars.get_tick_directions(df, direction_mode)

        # Asegurar columna dollar_value
        if "dollar_value" not in df.columns:
            # Calcular on-the-fly para ahorrar espacio en raw data
            df["dollar_value"] = df["price"] * df["quantity"]

        # Convertir a arrays numpy para Numba
        b_t = df["b_t"].values
        dollar_values = df["dollar_value"].values

        # Computar barras con Numba
        bar_indices = _compute_drbs_core(
            b_t, dollar_values, exp_lambda, init_exp_T, bb._get_min_ticks()
        )

        # Formar barras
        bars = []
        for start, end in bar_indices:
            bar = df.iloc[start:end]
            bars.append(self.base_bars._form_bar(bar))

        return pd.DataFrame(bars)

    def get_drbs_audited(
        self, df: pd.DataFrame, exp_lambda=0.9, init_exp_T=1600,
        direction_mode: str = "tick_rule",
    ) -> tuple[pd.DataFrame, BarAudit]:
        """Like ``get_drbs`` but returns (bars, BarAudit) with full tick accounting.

        ``audit.clean_ticks == audit.consumed_ticks + audit.residual_ticks`` is
        guaranteed by construction; the audit lets callers surface residual
        ticks instead of dropping them silently (``src/bars/bar_audit.py``).
        """
        df = df.copy()
        df["b_t"] = self.base_bars.get_tick_directions(df, direction_mode)

        if "dollar_value" not in df.columns:
            df["dollar_value"] = df["price"] * df["quantity"]

        b_t = df["b_t"].values
        dollar_values = df["dollar_value"].values

        bar_indices, residual = _compute_drbs_core_audited(
            b_t, dollar_values, exp_lambda, init_exp_T, bb._get_min_ticks()
        )

        bars = []
        for start, end in bar_indices:
            bar = df.iloc[start:end]
            bars.append(self.base_bars._form_bar(bar))
        bars_df = pd.DataFrame(bars)

        audit = BarAudit(
            clean_ticks=len(df),
            consumed_ticks=int(bars_df["n_ticks"].sum()) if len(bars_df) > 0 else 0,
            residual_ticks=residual,
            n_bars=len(bars_df),
        )
        audit.violations = check_bar_invariants(bars_df)

        return bars_df, audit
