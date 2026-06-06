from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    er_window: int = 10
    entry_trigger: float = 0.35
    regime_on: float = 0.30
    regime_off: float = 0.05
    profit_target: float = 0.03
    time_stop: int = 5
    trend_filter: int = 50


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779180130"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.er_window + 1, params.trend_filter)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        n = max(int(params.er_window), 1)

        ret = close.pct_change()
        # Net directional move over the window.
        net = close / close.shift(n) - 1.0
        # Path length: sum of absolute close-to-close returns over the window.
        path = ret.abs().rolling(n).sum()

        path_arr = path.to_numpy(dtype=float)
        net_arr = net.to_numpy(dtype=float)
        er_arr = np.zeros(len(close), dtype=float)
        valid = np.isfinite(path_arr) & np.isfinite(net_arr) & (path_arr > 1e-12)
        er_arr[valid] = net_arr[valid] / path_arr[valid]
        er_arr = np.clip(er_arr, -1.0, 1.0)
        er = pd.Series(er_arr, index=close.index)
        # Re-mark warmup region as NaN so the loop can skip it explicitly.
        er[~np.isfinite(path_arr)] = np.nan
        er[~np.isfinite(net_arr)] = np.nan

        sma = close.rolling(int(max(params.trend_filter, 1))).mean()

        out = pd.DataFrame(index=data.index)
        out["er"] = er
        out["sma"] = sma
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        er = indicators["er"].to_numpy(dtype=float)
        sma = indicators["sma"].to_numpy(dtype=float)
        n = len(close)

        above_trend = np.zeros(n, dtype=bool)
        for i in range(n):
            if np.isfinite(sma[i]) and close[i] > sma[i]:
                above_trend[i] = True

        signal = np.zeros(n, dtype=int)

        regime = False
        in_pos = False
        entry_price = 0.0
        bars_held = 0

        trigger = float(params.entry_trigger)
        on_th = float(params.regime_on)
        off_th = float(params.regime_off)
        pt = float(params.profit_target)
        tstop = int(max(params.time_stop, 1))

        for i in range(n):
            e = er[i]
            if not np.isfinite(e):
                # Stay flat through warmup; preserve any open position state.
                if in_pos:
                    signal[i] = 1
                continue

            # Hysteresis-latched trending regime: separate on / off thresholds.
            if not regime and e >= on_th:
                regime = True
            elif regime and e <= off_th:
                regime = False

            if in_pos:
                bars_held += 1
                gain = close[i] / entry_price - 1.0 if entry_price > 0.0 else 0.0
                if gain >= pt or bars_held >= tstop:
                    in_pos = False
                    bars_held = 0
                    entry_price = 0.0
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                prev_e = er[i - 1] if i >= 1 else np.nan
                two_bar_confirm = (
                    i >= 1
                    and np.isfinite(prev_e)
                    and e >= trigger
                    and prev_e >= trigger
                )
                if regime and above_trend[i] and two_bar_confirm:
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                    signal[i] = 1
                else:
                    signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
