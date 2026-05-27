from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class LotkaVolterraGapParams:
    population_window: int = 20
    gap_threshold_bps: float = 10.0
    min_population_diff: int = 2
    trend_ma: int = 50
    use_trend_filter: bool = True


class GeneratedStrategy(BaseStrategy[LotkaVolterraGapParams]):
    strategy_id = "gen_a1_1779654514"

    @classmethod
    def params_type(cls):
        return LotkaVolterraGapParams

    def warmup_bars(self, params: LotkaVolterraGapParams) -> int:
        return int(max(params.population_window, params.trend_ma)) + 2

    def indicators(self, data: pd.DataFrame, params: LotkaVolterraGapParams) -> pd.DataFrame:
        idx = data.index
        open_ = data["open"].astype(float)
        close = data["close"].astype(float)
        prior_close = close.shift(1)

        gap_bps = (open_ - prior_close) / prior_close.replace(0.0, np.nan) * 10000.0

        bull_token = (gap_bps >= params.gap_threshold_bps) & (close > open_)
        bear_token = (gap_bps <= -params.gap_threshold_bps) & (close < open_)

        bull_token_f = bull_token.fillna(False).astype(float)
        bear_token_f = bear_token.fillna(False).astype(float)

        bull_pop = bull_token_f.rolling(params.population_window, min_periods=params.population_window).sum()
        bear_pop = bear_token_f.rolling(params.population_window, min_periods=params.population_window).sum()

        ma = close.rolling(params.trend_ma, min_periods=params.trend_ma).mean()

        ind = pd.DataFrame(
            {
                "gap_bps": gap_bps,
                "bull_token": bull_token_f,
                "bear_token": bear_token_f,
                "bull_pop": bull_pop,
                "bear_pop": bear_pop,
                "ma": ma,
            },
            index=idx,
        )
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: LotkaVolterraGapParams,
    ) -> SignalFrame:
        close = data["close"].astype(float)

        bull_token = indicators["bull_token"].fillna(0.0).astype(bool)
        bear_token = indicators["bear_token"].fillna(0.0).astype(bool)
        bull_pop = indicators["bull_pop"]
        bear_pop = indicators["bear_pop"]
        ma = indicators["ma"]

        bull_minus_bear = (bull_pop - bear_pop).fillna(-1.0e9)
        bear_minus_bull = (bear_pop - bull_pop).fillna(-1.0e9)

        primitive_fresh_bull = bull_token
        primitive_pop_dominant = bull_minus_bear >= float(params.min_population_diff)

        if params.use_trend_filter:
            trend_ok = (close > ma).fillna(False)
        else:
            trend_ok = pd.Series(True, index=data.index)

        entry_cond = (primitive_fresh_bull & primitive_pop_dominant & trend_ok).fillna(False).values
        exit_cond = (bear_token | (bear_minus_bull >= float(params.min_population_diff))).fillna(False).values

        n = len(data)
        raw_signal = np.zeros(n, dtype=np.int64)
        in_pos = False
        for i in range(n):
            if in_pos:
                if bool(exit_cond[i]):
                    in_pos = False
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1
            else:
                if bool(entry_cond[i]):
                    in_pos = True
                    raw_signal[i] = 1
                else:
                    raw_signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
