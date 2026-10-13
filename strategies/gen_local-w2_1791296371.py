from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MonthBoundaryParams:
    tail: float = 0.20
    atr_k: float = 2.0


class GeneratedStrategy(BaseStrategy[MonthBoundaryParams]):
    strategy_id = "gen_local-w2_1791296371"

    @classmethod
    def params_type(cls) -> type[MonthBoundaryParams]:
        return MonthBoundaryParams

    @classmethod
    def warmup_bars(cls, params: MonthBoundaryParams) -> int:
        return 133

    def indicators(self, data: pd.DataFrame, params: MonthBoundaryParams) -> pd.DataFrame:
        prior_move = data["close"].pct_change(5).replace([np.inf, -np.inf], np.nan).shift(1)
        history = prior_move.shift(1)
        month = pd.Series(data.index.to_period("M"), index=data.index)
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        return pd.DataFrame({
            "prior_move": prior_move.fillna(0.0),
            "lower": history.rolling(126, min_periods=126).quantile(params.tail).fillna(0.0),
            "upper": history.rolling(126, min_periods=126).quantile(1.0 - params.tail).fillna(0.0),
            "atr": true_range.rolling(14, min_periods=14).mean().fillna(0.0),
            "first_session": month.ne(month.shift()),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MonthBoundaryParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        high_water = -np.inf
        low_water = np.inf
        stop = 0.0

        for i in range(len(data)):
            close = float(data["close"].iat[i])
            atr = float(indicators["atr"].iat[i])
            if position and i >= entry_bar:
                if position == 1:
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.atr_k * atr)
                    stopped = close <= stop
                else:
                    low_water = min(low_water, close)
                    stop = min(stop, low_water + params.atr_k * atr)
                    stopped = close >= stop
                if stopped or i - entry_bar >= 3:
                    position = 0
            elif i >= self.warmup_bars(params) and indicators["first_session"].iat[i] and atr > 0:
                move = float(indicators["prior_move"].iat[i])
                if move <= indicators["lower"].iat[i]:
                    position = 1
                    stop = -np.inf
                elif move >= indicators["upper"].iat[i]:
                    position = -1
                    stop = np.inf
                if position:
                    entry_bar = i + 1
                    high_water = -np.inf
                    low_water = np.inf
            signal[i] = position

        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    dates = pd.bdate_range("2020-01-01", periods=200)
    close = pd.Series(100.0 + 4.0 * np.sin(np.arange(200)), index=dates)
    sample = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1.0}, index=dates)
    strategy = GeneratedStrategy()
    params = MonthBoundaryParams()
    result = strategy.generate_signals(sample, strategy.indicators(sample, params), None, params)
    assert result.data["signal"].iloc[:134].eq(0).all()
    assert set(result.data["signal"].unique()) <= {-1, 0, 1}
    assert result.data["size"].gt(0).all()
