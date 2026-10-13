from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AutocorrelationParams:
    corr_window: int = 84
    min_corr: float = 0.10
    atr_mult: float = 3.0
    atr_window: int = 14
    hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[AutocorrelationParams]):
    strategy_id = "gen_local_1791301726"

    @classmethod
    def params_type(cls):
        return AutocorrelationParams

    def warmup_bars(self, params: AutocorrelationParams) -> int:
        return max(params.corr_window + 11, params.atr_window)

    def indicators(self, data: pd.DataFrame, params: AutocorrelationParams) -> pd.DataFrame:
        returns = data["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        for lag in (5, 10):
            correlation = returns.rolling(params.corr_window, min_periods=params.corr_window).corr(returns.shift(lag))
            out[f"corr_{lag}"] = correlation.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out[f"forecast_{lag}"] = (out[f"corr_{lag}"] * returns.shift(lag)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean().fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AutocorrelationParams,
    ) -> SignalFrame:
        ready = (indicators["corr_5"].abs() >= params.min_corr) & (indicators["corr_10"].abs() >= params.min_corr)
        forecasts_agree = indicators["forecast_5"] * indicators["forecast_10"] > 0
        votes = np.where(ready & forecasts_agree, np.sign(indicators["forecast_5"]), 0).astype(int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = prior_vote = age = 0
        extreme = stop = 0.0

        for i, (vote, close, atr) in enumerate(zip(votes, closes, atrs)):
            if position == 0:
                if vote != 0 and vote != prior_vote and atr > 0:
                    position = int(vote)
                    age = 0
            else:
                age += 1
                if age == 1:
                    extreme = close
                    stop = close - position * params.atr_mult * atr
                elif position == 1:
                    extreme = max(extreme, close)
                    stop = max(stop, extreme - params.atr_mult * atr)
                else:
                    extreme = min(extreme, close)
                    stop = min(stop, extreme + params.atr_mult * atr)
                if age >= params.hold_bars or (position == 1 and close <= stop) or (position == -1 and close >= stop):
                    position = 0
            raw[i] = position
            prior_vote = int(vote)

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
