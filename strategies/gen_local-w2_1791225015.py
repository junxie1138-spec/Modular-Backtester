from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    response_window: int = 24
    trail_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791225015"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.response_window + 2, 15)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        ret = close.div(close.shift(1)).sub(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prior_down = ret.shift(1).lt(0)
        window = params.response_window
        count = prior_down.rolling(window).sum().fillna(0.0)
        response = ret.where(prior_down, 0.0).rolling(window).sum().div(count.replace(0, np.nan)).fillna(0.0)
        shock = ret.abs().rolling(window).mean().fillna(0.0)
        prev_close = close.shift(1)
        true_range = pd.concat([data["high"] - data["low"], (data["high"] - prev_close).abs(), (data["low"] - prev_close).abs()], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["return"] = ret
        out["prior_return"] = ret.shift(1).fillna(0.0)
        out["response"] = response
        out["down_count"] = count
        out["shock"] = shock
        out["atr"] = true_range.rolling(14).mean().fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        valid = (indicators["down_count"] >= max(4, params.response_window // 4)) & (indicators["atr"] > 0)
        elastic = valid & (indicators["response"] > 0) & (indicators["return"] < -indicators["shock"])
        plastic = valid & (indicators["response"] <= 0) & (indicators["prior_return"] > 0) & (indicators["return"] > indicators["shock"])
        entry = (elastic | plastic).to_numpy()
        closes = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        raw = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        peak = 0.0
        stop = -np.inf
        for i in range(len(data)):
            if active:
                held += 1
                peak = max(peak, float(closes[i]))
                stop = max(stop, peak - params.trail_atr * float(atr[i]))
                if closes[i] <= stop or held >= 2:
                    active = False
            elif entry[i]:
                active = True
                held = 0
                peak = 0.0
                stop = -np.inf
            raw[i] = int(active)
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
