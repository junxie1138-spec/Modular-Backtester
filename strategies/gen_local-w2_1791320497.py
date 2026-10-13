from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    lookback: int = 8
    drawdown_atr: float = 1.25
    entry_balance: float = 0.15
    exit_balance: float = -0.65


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791320497"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    @staticmethod
    def warmup_bars(params: RecoveryParams) -> int:
        if not isinstance(params.lookback, int) or not 2 <= params.lookback <= 9:
            raise ValueError("lookback must be an integer from 2 through 9")
        values = [params.drawdown_atr, params.entry_balance, params.exit_balance]
        if not np.isfinite(values).all():
            raise ValueError("thresholds must be finite")
        if params.drawdown_atr <= 0:
            raise ValueError("drawdown_atr must be positive")
        if not -1 < params.exit_balance < 0 < params.entry_balance < 1:
            raise ValueError("require -1 < exit_balance < 0 < entry_balance < 1")
        return int(params.lookback + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        n = params.lookback
        close = data["close"]
        previous = close.shift(1)
        change = close.diff()
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        atr = true_range.rolling(n, min_periods=n).mean()
        upside = change.clip(lower=0).pow(2).rolling(n, min_periods=n).mean()
        downside = change.clip(upper=0).pow(2).rolling(n, min_periods=n).mean()
        energy = (upside + downside).replace(0.0, np.nan)
        peak = data["high"].rolling(n, min_periods=n).max()
        result = pd.DataFrame({
            "price_change": change,
            "true_range": true_range,
            "atr": atr,
            "upside_variance": upside,
            "downside_variance": downside,
            "total_variance": energy,
            "peak": peak,
            "drawdown_atr": (peak - close) / atr.replace(0.0, np.nan),
            "volatility_balance": (upside - downside) / energy,
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        balance = indicators["volatility_balance"].to_numpy(dtype=float)
        drawdown = indicators["drawdown_atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        recovering = False
        for i in range(len(data)):
            if np.isfinite(balance[i]):
                # Hysteresis retains the recovery condition between thresholds.
                if recovering:
                    if balance[i] <= params.exit_balance:
                        recovering = False
                elif (np.isfinite(drawdown[i])
                      and drawdown[i] >= params.drawdown_atr
                      and balance[i] >= params.entry_balance):
                    recovering = True
            signal[i] = int(recovering)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "volatility_balance": [0.3, 0.0, -0.7, 0.3, np.nan, -0.7],
        "drawdown_atr": [1.5, 0.0, 0.0, 1.5, np.nan, 0.0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1]
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) == 9
