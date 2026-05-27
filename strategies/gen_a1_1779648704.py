from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TidalAccelParams:
    roc_window: int = 3
    drawdown_lookback: int = 20
    drawdown_threshold: float = 0.03
    accel_amp_window: int = 50
    accel_amp_percentile: float = 0.70
    atr_window: int = 14
    breakeven_trigger: float = 0.01
    trail_atr_mult: float = 2.0
    max_hold_bars: int = 5
    trend_filter_ma: int = 200
    use_trend_filter: bool = True


class GeneratedStrategy(BaseStrategy[TidalAccelParams]):
    strategy_id = "gen_a1_1779648704"

    @classmethod
    def params_type(cls):
        return TidalAccelParams

    @classmethod
    def warmup_bars(cls, params: TidalAccelParams) -> int:
        return int(max(
            params.trend_filter_ma,
            params.accel_amp_window + params.roc_window + 2,
            params.drawdown_lookback,
            params.atr_window,
        )) + 5

    def indicators(self, data: pd.DataFrame, params: TidalAccelParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        roc = close.pct_change(params.roc_window)
        accel = roc.diff()

        rolling_high = close.rolling(params.drawdown_lookback, min_periods=1).max()
        drawdown = (close / rolling_high) - 1.0

        accel_abs = accel.abs()
        accel_amp_rank = accel_abs.rolling(
            params.accel_amp_window, min_periods=10
        ).rank(pct=True)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=1).mean()

        ma = close.rolling(params.trend_filter_ma, min_periods=1).mean()

        out = pd.DataFrame(
            {
                "roc": roc,
                "accel": accel,
                "drawdown": drawdown,
                "accel_amp_rank": accel_amp_rank,
                "atr": atr,
                "ma": ma,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TidalAccelParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy()
        accel = indicators["accel"].to_numpy()
        drawdown = indicators["drawdown"].to_numpy()
        accel_amp_rank = indicators["accel_amp_rank"].to_numpy()
        atr = indicators["atr"].to_numpy()
        ma = indicators["ma"].to_numpy()

        n = len(data)
        raw_signal = np.zeros(n, dtype=np.int64)

        in_position = False
        entry_price = 0.0
        bars_held = 0
        stop_price = 0.0
        breakeven_armed = False
        highest_close = 0.0

        for i in range(n):
            if not in_position:
                if i < 2:
                    continue
                a0 = accel[i]
                a1 = accel[i - 1]
                if not (np.isfinite(a0) and np.isfinite(a1)):
                    continue
                amp_now = accel_amp_rank[i]
                dd_now = drawdown[i]
                if not (np.isfinite(amp_now) and np.isfinite(dd_now)):
                    continue

                if params.use_trend_filter:
                    if not np.isfinite(ma[i]):
                        continue
                    trend_ok = close[i] >= ma[i]
                else:
                    trend_ok = True

                two_bar_accel = (a0 > 0.0) and (a1 > 0.0) and (a0 >= a1)
                antinode = amp_now >= params.accel_amp_percentile
                in_drawdown = dd_now <= -params.drawdown_threshold

                if two_bar_accel and antinode and in_drawdown and trend_ok:
                    in_position = True
                    raw_signal[i] = 1
                    entry_price = float(close[i])
                    highest_close = entry_price
                    bars_held = 0
                    breakeven_armed = False
                    a = float(atr[i]) if np.isfinite(atr[i]) else 0.0
                    stop_price = entry_price - params.trail_atr_mult * a
            else:
                bars_held += 1
                cur = float(close[i])
                if cur > highest_close:
                    highest_close = cur

                if (not breakeven_armed) and entry_price > 0.0:
                    gain = (cur / entry_price) - 1.0
                    if gain >= params.breakeven_trigger:
                        breakeven_armed = True
                        if stop_price < entry_price:
                            stop_price = entry_price

                a = float(atr[i]) if np.isfinite(atr[i]) else 0.0
                if breakeven_armed and a > 0.0:
                    trail_candidate = highest_close - params.trail_atr_mult * a
                    if trail_candidate > stop_price:
                        stop_price = trail_candidate

                exit_now = False
                if cur <= stop_price:
                    exit_now = True
                if bars_held >= params.max_hold_bars:
                    exit_now = True

                if exit_now:
                    raw_signal[i] = 0
                    in_position = False
                    entry_price = 0.0
                    bars_held = 0
                    breakeven_armed = False
                    highest_close = 0.0
                    stop_price = 0.0
                else:
                    raw_signal[i] = 1

        df = pd.DataFrame(
            {
                "signal": raw_signal,
                "size": np.ones(n, dtype=float),
            },
            index=data.index,
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
