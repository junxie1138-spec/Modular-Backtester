from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 60
    momentum_lookback: int = 20
    drawdown_lookback: int = 40
    rank_threshold: float = 0.5
    atr_period: int = 14
    breakeven_trigger: float = 0.03
    atr_trail_mult: float = 2.5
    max_hold_bars: int = 20
    base_size: float = 0.5


class GeneratedStrategy(BaseStrategy):
    strategy_id = "gen_a1_1779880930"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return max(
            params.rank_window + params.momentum_lookback + 2,
            params.rank_window + params.drawdown_lookback + 2,
            params.atr_period + 2,
        )

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        momentum = close.pct_change(params.momentum_lookback)

        rolling_max = close.rolling(params.drawdown_lookback, min_periods=1).max()
        drawdown = (close - rolling_max) / rolling_max

        predator_rank = momentum.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        prey_rank = (-drawdown).rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        predator_velocity = predator_rank.diff()
        prey_velocity = prey_rank.diff()

        pred_clip = predator_rank.clip(lower=0.0, upper=1.0)
        prey_clip = prey_rank.clip(lower=0.0, upper=1.0)
        signal_strength = np.sqrt(pred_clip * prey_clip)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        out = pd.DataFrame(
            {
                "predator_rank": predator_rank,
                "prey_rank": prey_rank,
                "predator_velocity": predator_velocity,
                "prey_velocity": prey_velocity,
                "signal_strength": signal_strength,
                "atr": atr,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        predator_rank = indicators["predator_rank"].to_numpy(dtype=float)
        prey_rank = indicators["prey_rank"].to_numpy(dtype=float)
        predator_velocity = indicators["predator_velocity"].to_numpy(dtype=float)
        prey_velocity = indicators["prey_velocity"].to_numpy(dtype=float)
        signal_strength = indicators["signal_strength"].to_numpy(dtype=float)

        n = len(data)
        raw_signal = np.zeros(n, dtype=np.int64)
        size_arr = np.zeros(n, dtype=np.float64)

        in_position = False
        entry_price = 0.0
        breakeven_armed = False
        trail_stop = -np.inf
        bars_held = 0
        position_size = 0.0

        for i in range(n):
            pr = predator_rank[i]
            qr = prey_rank[i]
            pv = predator_velocity[i]
            qv = prey_velocity[i]
            a = atr[i]
            ss = signal_strength[i]

            if (
                np.isnan(pr)
                or np.isnan(qr)
                or np.isnan(pv)
                or np.isnan(qv)
                or np.isnan(a)
                or np.isnan(ss)
            ):
                raw_signal[i] = 0
                size_arr[i] = 0.0
                continue

            if not in_position:
                expansion = (
                    pr > params.rank_threshold
                    and qr > params.rank_threshold
                    and pv > 0.0
                    and qv > 0.0
                    and ss > 0.0
                )
                if expansion:
                    in_position = True
                    entry_price = close[i]
                    breakeven_armed = False
                    trail_stop = -np.inf
                    bars_held = 0
                    scaled = params.base_size + (1.0 - params.base_size) * ss
                    if scaled > 1.0:
                        scaled = 1.0
                    if scaled < 0.05:
                        scaled = 0.05
                    position_size = float(scaled)
                    raw_signal[i] = 1
                    size_arr[i] = position_size
                else:
                    raw_signal[i] = 0
                    size_arr[i] = 0.0
            else:
                bars_held += 1
                if not breakeven_armed:
                    if close[i] >= entry_price * (1.0 + params.breakeven_trigger):
                        breakeven_armed = True
                        trail_stop = entry_price

                if breakeven_armed:
                    candidate = close[i] - params.atr_trail_mult * a
                    if candidate > trail_stop:
                        trail_stop = candidate

                exit_now = False
                if breakeven_armed and low[i] <= trail_stop:
                    exit_now = True
                if bars_held >= params.max_hold_bars:
                    exit_now = True

                if exit_now:
                    raw_signal[i] = 0
                    size_arr[i] = 0.0
                    in_position = False
                    entry_price = 0.0
                    breakeven_armed = False
                    trail_stop = -np.inf
                    bars_held = 0
                    position_size = 0.0
                else:
                    raw_signal[i] = 1
                    size_arr[i] = position_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        size_series = pd.Series(size_arr, index=data.index).shift(1).fillna(0.0)
        size_series = size_series.where(size_series > 0.0, 1.0)
        df["size"] = size_series.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
