from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlasticCoilParams:
    atr_window: int = 14
    rank_window: int = 100
    compress_q: float = 0.25
    release_window: int = 6
    expansion_mult: float = 1.3
    mid_window: int = 20
    set_lag: int = 20
    set_thresh: float = 0.01
    ma_window: int = 200
    init_stop_mult: float = 2.0
    breakeven_pct: float = 0.02
    trail_mult: float = 2.5
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[PlasticCoilParams]):
    strategy_id = "gen_a1_1778904263"

    @classmethod
    def params_type(cls) -> type[PlasticCoilParams]:
        return PlasticCoilParams

    @staticmethod
    def warmup_bars(params: PlasticCoilParams) -> int:
        return int(max(
            params.ma_window,
            params.atr_window + params.rank_window,
            params.mid_window + params.set_lag,
        )) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: PlasticCoilParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        atr_floor = atr.rolling(
            params.rank_window, min_periods=params.rank_window
        ).quantile(params.compress_q)

        compressed = (atr < atr_floor).astype(float)
        compressed_recently = (
            compressed.shift(1)
            .rolling(params.release_window, min_periods=1)
            .max()
        )

        expansion_bar = tr > (params.expansion_mult * atr.shift(1))

        mid = close.rolling(params.mid_window, min_periods=params.mid_window).mean()
        mid_prev = mid.shift(params.set_lag)
        plastic_up = mid > (mid_prev * (1.0 + params.set_thresh))

        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        regime = close > ma
        direction_up = close > mid

        entry = (
            (compressed_recently >= 1.0)
            & expansion_bar
            & plastic_up
            & regime
            & direction_up
        )

        out = pd.DataFrame(index=data.index)
        out["tr"] = tr
        out["atr"] = atr
        out["atr_floor"] = atr_floor
        out["mid"] = mid
        out["ma"] = ma
        out["entry"] = entry.fillna(False).astype(bool)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PlasticCoilParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].to_numpy()
        n = len(close)

        pos = np.zeros(n, dtype=np.int64)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        breakeven_done = False
        bars_held = 0

        for i in range(n):
            if not in_pos:
                a = atr[i]
                if bool(entry[i]) and np.isfinite(a) and a > 0.0:
                    in_pos = True
                    entry_price = close[i]
                    stop = entry_price - params.init_stop_mult * a
                    breakeven_done = False
                    bars_held = 0
                    pos[i] = 1
            else:
                bars_held += 1
                if (not breakeven_done) and high[i] >= entry_price * (
                    1.0 + params.breakeven_pct
                ):
                    if entry_price > stop:
                        stop = entry_price
                    breakeven_done = True
                if breakeven_done:
                    a = atr[i]
                    if np.isfinite(a):
                        trail = close[i] - params.trail_mult * a
                        if trail > stop:
                            stop = trail
                if (close[i] <= stop) or (bars_held >= params.max_hold):
                    in_pos = False
                    pos[i] = 0
                else:
                    pos[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
