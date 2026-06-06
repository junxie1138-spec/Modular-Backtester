from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    vol_ma_len: int = 20
    vol_surge_mult: float = 1.3
    obv_ma_len: int = 50
    profit_target: float = 0.08
    time_stop: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778903485"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.vol_ma_len, params.obv_ma_len)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        ind = pd.DataFrame(index=data.index)

        # On-balance-volume style accumulation/distribution line.
        direction = np.sign(close.diff()).fillna(0.0)
        obv = (direction * volume).cumsum()
        obv_ma = obv.rolling(params.obv_ma_len, min_periods=params.obv_ma_len).mean()

        # Accumulation regime: OBV trending above its own moving average.
        regime = (obv > obv_ma).fillna(False)

        # Volume-confirmed up-bar: close up AND volume above surge threshold.
        vol_ma = volume.rolling(params.vol_ma_len, min_periods=params.vol_ma_len).mean()
        vol_surge = (volume > (vol_ma * params.vol_surge_mult)).fillna(False)
        up_bar = (close > close.shift(1)).fillna(False)
        confirmed_up = up_bar & vol_surge

        # Two-bar confirmation: two consecutive volume-confirmed up-bars.
        two_bar = confirmed_up & confirmed_up.shift(1).fillna(False)

        entry_signal = (regime & two_bar).astype(float)

        ind["obv"] = obv
        ind["obv_ma"] = obv_ma
        ind["regime"] = regime.astype(float)
        ind["vol_ma"] = vol_ma
        ind["confirmed_up"] = confirmed_up.astype(float)
        ind["entry_signal"] = entry_signal
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        entry = indicators["entry_signal"].fillna(0.0).to_numpy(dtype=float) > 0.5
        n = len(close)

        pos = np.zeros(n, dtype=np.int64)
        in_pos = False
        entry_price = 0.0
        bars_held = 0

        profit_target = float(params.profit_target)
        time_stop = int(params.time_stop)

        for i in range(n):
            if in_pos:
                bars_held += 1
                ret = (close[i] / entry_price) - 1.0 if entry_price > 0.0 else 0.0
                if ret >= profit_target or bars_held >= time_stop:
                    in_pos = False
                    entry_price = 0.0
                    bars_held = 0
                    pos[i] = 0
                else:
                    pos[i] = 1
            else:
                if entry[i]:
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                    pos[i] = 1
                else:
                    pos[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
