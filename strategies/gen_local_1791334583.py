from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    ma_window: int = 20
    volume_window: int = 6
    atr_window: int = 14
    entry_z: float = 1.5
    reset_z: float = 0.5
    participation_gain: float = 0.03
    trail_atr: float = 2.0
    max_hold: int = 10

    def __post_init__(self):
        if min(self.ma_window, self.volume_window, self.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 <= self.reset_z < self.entry_z and 0 <= self.participation_gain <= 1):
            raise ValueError("Invalid entry thresholds.")
        if not (np.isfinite(self.trail_atr) and self.trail_atr > 0 and self.max_hold > 0):
            raise ValueError("Invalid exit parameters.")


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = "gen_local_1791334583"

    @classmethod
    def params_type(cls):
        return ParticipationParams

    def warmup_bars(self, params):
        return int(max(params.ma_window, 2 * params.volume_window, params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].replace(0, np.nan)
        volume = data["volume"].where(data["volume"] >= 0)
        out["volume_sum"] = volume.rolling(params.volume_window, min_periods=params.volume_window).sum()
        out["volume_square_sum"] = volume.pow(2).rolling(params.volume_window, min_periods=params.volume_window).sum()
        out["participation"] = out["volume_sum"].pow(2) / (
            params.volume_window * out["volume_square_sum"].replace(0, np.nan)
        )
        out["previous_participation"] = out["participation"].shift(params.volume_window)
        out["participation_change"] = out["participation"] - out["previous_participation"]
        out["previous_close"] = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["tr"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        active = False
        armed = True
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        rows = zip(data["close"].to_numpy(), indicators["z"].to_numpy(),
                   indicators["participation_change"].to_numpy(), indicators["atr"].to_numpy())
        for i, (close, z, gain, atr) in enumerate(rows):
            if not (np.isfinite(close) and close > 0 and np.isfinite(z)
                    and np.isfinite(gain) and np.isfinite(atr) and atr > 0):
                active = False
                continue
            if active:
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or i - entry_bar + 1 >= params.max_hold:
                    active = False
            else:
                if z <= params.reset_z:
                    armed = True
                if armed and z >= params.entry_z and gain >= params.participation_gain:
                    active = True
                    armed = False
                    entry_bar = i + 1
                    high_water = -np.inf
                    stop = -np.inf
            signals[i] = int(active)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 101., 104., 103., 102., 101.]}, index=index)
    ind = pd.DataFrame({"z": [0., 2., 2., 2., 2., 0.],
                        "participation_change": [0.1] * 6,
                        "atr": [1., 1., 1., 10., 1., 1.]}, index=index)
    params = ParticipationParams()
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0]
    ind["participation_change"] = 0.0
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert not result["signal"].any()


if __name__ == "__main__":
    _self_check()
