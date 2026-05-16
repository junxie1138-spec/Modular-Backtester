from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    vol_ma_len: int = 20
    vol_mult: float = 1.3
    spike_mult: float = 2.5
    refractory_bars: int = 5
    regime_window: int = 15
    regime_threshold: int = 3
    ma_len: int = 200
    atr_len: int = 14
    atr_mult: float = 2.5
    max_hold: int = 12


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778893620"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.ma_len + params.regime_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        ind = pd.DataFrame(index=data.index)

        vol_ma = volume.rolling(
            params.vol_ma_len, min_periods=params.vol_ma_len
        ).mean()
        ret = close.diff()

        confirmed = volume > (params.vol_mult * vol_ma)
        up_conf = (ret > 0) & confirmed
        dn_conf = (ret < 0) & confirmed

        net_conf = (
            up_conf.astype(float) - dn_conf.astype(float)
        ).rolling(params.regime_window, min_periods=params.regime_window).sum()

        ma = close.rolling(params.ma_len, min_periods=params.ma_len).mean()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()

        ind["up_conf"] = up_conf.astype(float)
        ind["net_conf"] = net_conf
        ind["ma"] = ma
        ind["atr"] = atr
        ind["spike"] = (volume > (params.spike_mult * vol_ma)).astype(float)
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        up_conf = indicators["up_conf"].to_numpy(dtype=float)
        net_conf = indicators["net_conf"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        spike = indicators["spike"].to_numpy(dtype=float)

        n = len(close)
        signal = np.zeros(n, dtype=np.int64)

        position = 0
        high_water = 0.0
        hold = 0
        refractory = 0

        for i in range(n):
            if refractory > 0:
                refractory -= 1
            if spike[i] == 1.0:
                refractory = int(params.refractory_bars)

            if position == 0:
                bull = (not np.isnan(ma[i])) and close[i] > ma[i]
                regime = (
                    not np.isnan(net_conf[i])
                ) and net_conf[i] >= params.regime_threshold
                trigger = up_conf[i] == 1.0
                atr_ok = (not np.isnan(atr[i])) and atr[i] > 0.0
                if bull and regime and trigger and atr_ok and refractory == 0:
                    position = 1
                    high_water = close[i]
                    hold = 0
                    signal[i] = 1
            else:
                hold += 1
                if close[i] > high_water:
                    high_water = close[i]
                stop = high_water - params.atr_mult * atr[i]
                stop_hit = (not np.isnan(atr[i])) and close[i] < stop
                if stop_hit or hold >= params.max_hold:
                    position = 0
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
