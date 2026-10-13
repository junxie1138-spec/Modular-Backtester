from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    lookback: int = 20
    volume_window: int = 20
    trend_window: int = 50
    atr_period: int = 14
    score_threshold: float = 0.35
    stop_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791225232"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.trend_window, params.volume_window + params.lookback, params.atr_period + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean().fillna(0.0)

        baseline = data["volume"].shift(1).rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean().replace(0, np.nan)
        relative_volume = data["volume"].div(baseline)
        weight = relative_volume.where(relative_volume > 1.0, 0.0).clip(upper=3.0).fillna(0.0)
        returns = data["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        net = (returns * weight).rolling(params.lookback, min_periods=params.lookback).sum()
        gross = (returns.abs() * weight).rolling(params.lookback, min_periods=params.lookback).sum()
        out["score"] = net.div(gross.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        out["confirmed_days"] = (weight > 0).rolling(
            params.lookback, min_periods=params.lookback
        ).sum().fillna(0.0)
        average = data["close"].rolling(params.trend_window, min_periods=params.trend_window).mean()
        out["uptrend"] = (data["close"] > average).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        score = indicators["score"].to_numpy()
        days = indicators["confirmed_days"].to_numpy()
        uptrend = indicators["uptrend"].to_numpy()
        atr = indicators["atr"].to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        signal = np.zeros(len(data), dtype=int)
        size = np.ones(len(data), dtype=float)
        active = False
        pending_entry = False
        entry_atr = 0.0
        stop_price = 0.0
        held_size = 1.0
        held_bars = 0

        for i in range(len(data)):
            if pending_entry:
                stop_price = float(opens[i]) - params.stop_atr * entry_atr
                active = True
                pending_entry = False
                held_bars = 0

            if active:
                held_bars += 1
                if closes[i] < stop_price or held_bars >= params.max_hold:
                    active = False
                else:
                    signal[i] = 1
                    size[i] = held_size
            elif (
                atr[i] > 0
                and uptrend[i]
                and days[i] >= max(4, params.lookback // 4)
                and score[i] > params.score_threshold
                and i > 0
                and score[i] > score[i - 1]
            ):
                strength = (score[i] - params.score_threshold) / (1.0 - params.score_threshold)
                held_size = float(np.clip(0.25 + 0.75 * strength, 0.25, 1.0))
                entry_atr = float(atr[i])
                pending_entry = True
                signal[i] = 1
                size[i] = held_size

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
