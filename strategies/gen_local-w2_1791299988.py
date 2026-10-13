from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RetestParams:
    ma_window: int = 20
    rank_window: int = 126
    retest_bars: int = 7
    tail_quantile: float = 0.15
    profit_target: float = 0.03
    hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[RetestParams]):
    strategy_id = "gen_local-w2_1791299988"

    @classmethod
    def params_type(cls):
        return RetestParams

    def warmup_bars(self, params: RetestParams) -> int:
        return params.ma_window + max(params.rank_window, params.retest_bars) + 1

    def indicators(self, data: pd.DataFrame, params: RetestParams) -> pd.DataFrame:
        close = data["close"]
        mean = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        scale = close.rolling(params.ma_window, min_periods=params.ma_window).std().replace(0, np.nan)
        z = (close - mean) / scale
        lower = z.rolling(params.rank_window, min_periods=params.rank_window).quantile(params.tail_quantile).shift(1)
        upper = z.rolling(params.rank_window, min_periods=params.rank_window).quantile(1 - params.tail_quantile).shift(1)
        prior_low_z = z.rolling(params.retest_bars, min_periods=params.retest_bars).min().shift(1)
        prior_high_z = z.rolling(params.retest_bars, min_periods=params.retest_bars).max().shift(1)
        prior_low = data["low"].rolling(params.retest_bars, min_periods=params.retest_bars).min().shift(1)
        prior_high = data["high"].rolling(params.retest_bars, min_periods=params.retest_bars).max().shift(1)
        out = pd.DataFrame(index=data.index)
        out["long_entry"] = (prior_low_z.lt(lower) & data["low"].lt(prior_low) & z.gt(prior_low_z) & close.gt(data["open"])).fillna(False)
        out["short_entry"] = (prior_high_z.gt(upper) & data["high"].gt(prior_high) & z.lt(prior_high_z) & close.lt(data["open"])).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RetestParams,
    ) -> SignalFrame:
        position = 0
        entry_bar = -1
        signals = []
        for i in range(len(data)):
            if position and i >= entry_bar:
                entry_price = float(data["open"].iloc[entry_bar])
                gain = position * (float(data["close"].iloc[i]) / entry_price - 1)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                    position = 0
            elif not position:
                long_entry = bool(indicators["long_entry"].iloc[i])
                short_entry = bool(indicators["short_entry"].iloc[i])
                if long_entry != short_entry:
                    position = 1 if long_entry else -1
                    entry_bar = i + 1
            signals.append(position)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
