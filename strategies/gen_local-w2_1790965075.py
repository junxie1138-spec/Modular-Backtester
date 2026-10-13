from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownDryupParams:
    drawdown_window: int = 6
    min_drawdown: float = 0.009
    max_volume_ratio: float = 0.90


class GeneratedStrategy(BaseStrategy[DrawdownDryupParams]):
    strategy_id = "gen_local-w2_1790965075"

    @classmethod
    def params_type(cls):
        return DrawdownDryupParams

    def warmup_bars(self, params):
        return int(max(params.drawdown_window, 18))

    def indicators(self, data, params):
        if params.drawdown_window < 2 or params.min_drawdown <= 0 or params.max_volume_ratio <= 0:
            raise ValueError("Require drawdown_window >= 2 and positive thresholds")
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        volume = data["volume"].replace([np.inf, -np.inf], np.nan)
        volume = volume.where(volume >= 0)
        out = pd.DataFrame(index=data.index)
        out["recent_peak"] = close.rolling(params.drawdown_window, min_periods=params.drawdown_window).max()
        out["drawdown"] = 1.0 - close / out["recent_peak"]
        out["recent_volume"] = volume.rolling(3, min_periods=3).mean()
        out["prior_volume"] = volume.shift(3).rolling(15, min_periods=15).mean()
        out["volume_ratio"] = out["recent_volume"] / out["prior_volume"].where(out["prior_volume"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        agree = (indicators["drawdown"] >= params.min_drawdown) & (indicators["volume_ratio"] <= params.max_volume_ratio)
        df = pd.DataFrame(index=data.index)
        # Both primitives must remain true; their AND reversal is the only exit.
        df["signal"] = agree.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = DrawdownDryupParams()
    strategy = GeneratedStrategy()
    close = np.array([100.0] * 24 + [99.0] * 7)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": [100.0] * 24 + [50.0] * 7}, index=pd.date_range("2020-01-01", periods=31, freq="B"))
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0] * 25 + [1] * 5 + [0]
    no_dryup = data.assign(volume=100.0)
    assert not strategy.generate_signals(no_dryup, strategy.indicators(no_dryup, params), ctx, params).data["signal"].any()
    assert signals["size"].gt(0).all()
