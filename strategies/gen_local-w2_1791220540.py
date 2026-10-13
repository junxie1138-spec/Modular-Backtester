from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    drawdown_atr: float = 1.5
    entry_margin: float = 0.2
    exit_margin: float = -0.5


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791220540"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if params.drawdown_atr <= 0 or params.exit_margin >= params.entry_margin:
            raise ValueError("Require positive drawdown_atr and exit_margin < entry_margin.")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr_fast"] = out["true_range"].rolling(2, min_periods=2).mean()
        out["atr_slow"] = out["true_range"].rolling(8, min_periods=8).mean()
        out["prior_peak"] = data["close"].rolling(4, min_periods=4).max().shift(4)
        out["recent_trough"] = data["close"].rolling(4, min_periods=4).min()
        denominator = out["atr_slow"].where(out["atr_slow"] > 0)
        out["drawdown_depth"] = (out["prior_peak"] - out["recent_trough"]) / denominator
        out["recovery_margin"] = (
            data["close"] - out["recent_trough"] - out["atr_fast"]
        ) / denominator
        out["score"] = np.minimum(
            out["drawdown_depth"] - params.drawdown_atr,
            out["recovery_margin"],
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RecoveryParams,
    ) -> SignalFrame:
        active = False
        signals = np.zeros(len(data), dtype=int)
        for i, score in enumerate(indicators["score"].to_numpy()):
            if not np.isfinite(score):
                active = False
            else:
                # The same recovery condition uses hysteresis to reverse.
                threshold = params.exit_margin if active else params.entry_margin
                active = bool(score > threshold)
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    scores = pd.DataFrame(
        {"score": [np.nan, 0.3, 0.0, -0.4, -0.6, 0.0, 0.4, -0.5, np.nan]},
        index=index,
    )
    result = strategy.generate_signals(data, scores, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 1, 0]
    assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) == 9
    actual = strategy.indicators(data, params)
    assert strategy.generate_signals(data, actual, ctx, params).data["signal"].eq(0).all()
