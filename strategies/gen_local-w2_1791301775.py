from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoilParams:
    shock_threshold: float = 1.8
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[RecoilParams]):
    strategy_id = "gen_local-w2_1791301775"

    @classmethod
    def params_type(cls):
        return RecoilParams

    @classmethod
    def warmup_bars(cls, params: RecoilParams) -> int:
        return 22

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: RecoilParams) -> pd.DataFrame:
        ret = data["close"].pct_change()
        shock = ret.shift(1).rolling(3).sum()
        volatility = ret.rolling(20).std().shift(1)
        recoil = (-ret / shock.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        return pd.DataFrame({
            "shock": shock.fillna(0.0),
            "volatility": volatility.fillna(0.0),
            "recoil": recoil.fillna(0.0),
            "atr": true_range.rolling(14).mean().fillna(0.0),
        }, index=data.index)

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoilParams,
    ) -> SignalFrame:
        shock = indicators["shock"].to_numpy()
        volatility = indicators["volatility"].to_numpy()
        recoil = indicators["recoil"].to_numpy()
        atr = indicators["atr"].to_numpy()
        opens = data["open"].to_numpy()
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()
        closes = data["close"].to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = None
        stop = None
        armed = False
        age = 0

        for i in range(len(data)):
            if position:
                if entry_price is None:
                    entry_price = opens[i]
                    stop = entry_price - position * params.trail_atr * atr[i - 1]
                age += 1
                stopped = closes[i] <= stop if position == 1 else closes[i] >= stop
                if stopped or age >= 10:
                    position = 0
                    entry_price = None
                    stop = None
                    armed = False
                    age = 0
                elif position == 1:
                    armed = armed or highs[i] >= entry_price * 1.01
                    if armed:
                        stop = max(stop, entry_price, highs[i] - params.trail_atr * atr[i])
                else:
                    armed = armed or lows[i] <= entry_price * 0.99
                    if armed:
                        stop = min(stop, entry_price, lows[i] + params.trail_atr * atr[i])
            elif (i >= cls.warmup_bars(params) and volatility[i] > 0
                  and atr[i] > 0 and 0.20 <= recoil[i] <= 0.60):
                limit = params.shock_threshold * volatility[i] * np.sqrt(3)
                if shock[i] < -limit:
                    position = 1
                elif shock[i] > limit:
                    position = -1
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
