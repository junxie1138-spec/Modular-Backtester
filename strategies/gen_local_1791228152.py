from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    complexity_window: int = 12
    complexity_threshold: float = 0.85
    channel_window: int = 20
    edge_fraction: float = 0.40
    atr_window: int = 14
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791228152"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.channel_window, params.atr_window + 1,
                       params.complexity_window + 3))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"]
        a, b, c = close.shift(2), close.shift(1), close
        patterns = [
            (a < b) & (b < c),
            (a < c) & (c < b),
            (b < a) & (a < c),
            (b < c) & (c < a),
            (c < a) & (a < b),
            (c < b) & (b < a),
        ]
        ordinal = pd.Series(np.select(patterns, range(6), default=np.nan),
                            index=data.index)
        probabilities = pd.concat([
            ordinal.eq(k).astype(float).where(ordinal.notna()).rolling(
                params.complexity_window,
                min_periods=params.complexity_window
            ).mean() for k in range(6)
        ], axis=1)
        complexity = -(probabilities * np.log(
            probabilities.clip(lower=1e-12)
        )).sum(axis=1, min_count=6) / np.log(6.0)
        channel_low = data["low"].rolling(params.channel_window).min()
        channel_high = data["high"].rolling(params.channel_window).max()
        width = (channel_high - channel_low).replace(0.0, np.nan)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        return pd.DataFrame({
            "ordinal": ordinal,
            "complexity": complexity,
            "relative_position": (close - channel_low) / width,
            "ma200": close.rolling(200, min_periods=200).mean(),
            "atr": true_range.rolling(params.atr_window).mean(),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        ready = pd.Series(np.isfinite(indicators.to_numpy()).all(axis=1),
                          index=data.index)
        collapse = (indicators["complexity"] <= params.complexity_threshold) & (
            indicators["complexity"].shift(1) > params.complexity_threshold)
        long_entry = ready & collapse & (data["close"] > indicators["ma200"]) & (
            indicators["relative_position"] <= params.edge_fraction) & (
            indicators["ordinal"] == 0)
        short_entry = ready & collapse & (data["close"] < indicators["ma200"]) & (
            indicators["relative_position"] >= 1.0 - params.edge_fraction) & (
            indicators["ordinal"] == 5)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_atr = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                if entry_bar < 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                # The entry ATR and entry price stay fixed for the entire trade.
                stopped = np.isfinite(closes[i]) and (
                    position * (closes[i] - entry_price) <= -params.stop_atr * entry_atr)
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    position = 0
                    entry_bar = -1
                signals[i] = position
                continue
            if candidates[i] and np.isfinite(closes[i]) and closes[i] > 0 and atr[i] > 0:
                position = int(candidates[i])
                entry_atr = atr[i]
                entry_bar = -1
                signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=212, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[202], "close"] = 95.0
    features = pd.DataFrame({"ordinal": 0.0, "complexity": 0.95,
                             "relative_position": 0.2, "ma200": 90.0,
                             "atr": 2.0}, index=index)
    features.loc[index[[200, 205]], "complexity"] = 0.7
    strategy = GeneratedStrategy()
    params = StrategyParams()
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].iloc[[200, 201, 202, 203, 205, 206, 209, 210]].tolist() == [0, 1, 1, 0, 0, 1, 1, 0]
    assert result["signal"].isin([-1, 0, 1]).all() and result["size"].gt(0).all()
    assert strategy.indicators(data, params).index.equals(data.index)
    assert strategy.warmup_bars(params) >= 200
