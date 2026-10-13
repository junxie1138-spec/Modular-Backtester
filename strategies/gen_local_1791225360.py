from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    drawdown_window: int = 63
    energy_window: int = 5
    shock_window: int = 15
    decay_ratio: float = 0.55
    min_drawdown: float = 0.035
    hold_bars: int = 16


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791225360"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.drawdown_window, 16 + params.energy_window + params.shock_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data["close"]
        returns = np.log(close.where(close > 0)).diff()
        out = pd.DataFrame(index=data.index)
        out["drawdown"] = close / close.rolling(params.drawdown_window, min_periods=params.drawdown_window).max() - 1
        out["rebound"] = close / data["low"].rolling(10, min_periods=10).min() - 1
        total = pd.Series(0.0, index=data.index)
        for scale in (4, 8, 16):
            half = scale // 2
            recent = returns.rolling(half, min_periods=half).sum()
            coefficient = (recent - recent.shift(half)) / np.sqrt(scale)
            energy = coefficient.pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
            out[f"energy_{scale}"] = energy
            total = total + energy
        out["total_energy"] = total
        out["shock_peak"] = total.rolling(params.shock_window, min_periods=params.shock_window).max()
        out["energy_ratio"] = total / out["shock_peak"].where(out["shock_peak"] > 0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        change = data["close"].pct_change()
        up_two = change.gt(0) & change.shift(1).gt(0)
        down_two = change.lt(0) & change.shift(1).lt(0)
        drawn = indicators["drawdown"].le(-params.min_drawdown)
        shocked = indicators["shock_peak"].gt(0)
        long_entry = drawn & shocked & indicators["rebound"].ge(0.01) & indicators["energy_ratio"].le(params.decay_ratio) & up_two
        short_entry = drawn & shocked & indicators["rebound"].shift(2).ge(0.015) & indicators["energy_ratio"].ge(0.8) & indicators["energy_4"].gt(1.5 * indicators["energy_4"].shift(2)) & down_two
        longs = long_entry.fillna(False).to_numpy(dtype=bool)
        shorts = short_entry.fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if position and i == exit_at:
                position = 0
            elif not position and i >= warmup:
                if longs[i]:
                    position = 1
                    exit_at = i + params.hold_bars
                elif shorts[i]:
                    position = -1
                    exit_at = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RecoveryParams(hold_bars=3)
    close = np.full(80, 100.0)
    close[63] = 101.0
    close[64:] = 102.0
    data = pd.DataFrame({"close": close})
    indicators = pd.DataFrame({"drawdown": -0.05, "rebound": 0.02, "shock_peak": 1.0, "energy_ratio": 0.4, "energy_4": 0.1}, index=data.index)
    signals = GeneratedStrategy().generate_signals(data, indicators, None, params).data["signal"]
    assert signals.iloc[64:69].tolist() == [0, 1, 1, 1, 0]
