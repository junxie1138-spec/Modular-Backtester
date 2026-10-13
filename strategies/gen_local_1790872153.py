from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    ma_window: int = 60
    impact_window: int = 10
    recovery_z: float = -1.5
    impact_ratio: float = 0.8
    holding_bars: int = 18

    def __post_init__(self):
        if self.ma_window < 2 or self.impact_window < 4 or self.holding_bars < 1:
            raise ValueError("Invalid lookback or holding period")
        if not np.isfinite(self.recovery_z) or self.recovery_z >= 0:
            raise ValueError("recovery_z must be finite and negative")
        if not np.isfinite(self.impact_ratio) or self.impact_ratio <= 0:
            raise ValueError("impact_ratio must be finite and positive")


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790872153"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params):
        return int(max(params.ma_window + 1, params.impact_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0).replace(0, np.nan)
        out["z"] = (close - out["ma"]) / out["std"]
        out["previous_z"] = out["z"].shift(1)
        out["return"] = (close / close.shift(1) - 1).replace([np.inf, -np.inf], np.nan)
        out["turnover"] = close * data["volume"].where(data["volume"] > 0)
        valid = out["return"].notna() & out["turnover"].notna()
        for side, mask in (("down", out["return"] < 0), ("up", out["return"] > 0)):
            out[side + "_move"] = out["return"].abs().where(mask, 0.0).where(valid)
            out[side + "_turnover"] = out["turnover"].where(mask, 0.0).where(valid)
            out[side + "_count"] = mask.astype(float).where(valid).rolling(params.impact_window).sum()
            out[side + "_move_sum"] = out[side + "_move"].rolling(params.impact_window).sum()
            out[side + "_turnover_sum"] = out[side + "_turnover"].rolling(params.impact_window).sum()
            out[side + "_impact"] = out[side + "_move_sum"] / out[side + "_turnover_sum"].replace(0, np.nan)
        out["impact_balance"] = out["down_impact"] / out["up_impact"].replace(0, np.nan)
        out["impact_balance"] = out["impact_balance"].where(
            (out["down_count"] >= 2) & (out["up_count"] >= 2)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        recovery = (
            (indicators["previous_z"] <= params.recovery_z)
            & (indicators["z"] > params.recovery_z)
            & (indicators["z"] < 0)
        )
        absorption = indicators["impact_balance"] <= params.impact_ratio
        entries = (recovery & absorption).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(entries):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    exit_at = -1
            elif enter:
                signal[i] = 1
                exit_at = i + params.holding_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RecoveryParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=40)
    data = pd.DataFrame(index=index)
    z = pd.Series(-1.0, index=index)
    z.iloc[:2] = -2.0
    indicators = pd.DataFrame(
        {"z": z, "previous_z": z.shift(1), "impact_balance": 0.5}, index=index
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=61)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert np.array_equal(np.flatnonzero(result["signal"]), np.arange(3, 21))
    indicators["impact_balance"] = 2.0
    assert not strategy.generate_signals(data, indicators, ctx, params).data["signal"].any()
