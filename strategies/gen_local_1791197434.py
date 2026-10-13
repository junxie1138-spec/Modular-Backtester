from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    volume_ratio: float = 1.5
    trail_atr: float = 1.4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791197434"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 24

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.volume_ratio) or params.volume_ratio <= 1:
            raise ValueError("volume_ratio must exceed one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive")
        close, volume = data["close"], data["volume"]
        prev = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prev).abs(),
            (data["low"] - prev).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        baseline = volume.shift(1).rolling(20, min_periods=20).mean()
        spike = (
            (volume > params.volume_ratio * baseline)
            & (close - prev > 0.5 * atr)
            & (close > data["open"])
        )
        floor = ((data["open"] + close) / 2).shift(3)
        digestion = (
            (volume.shift(1) < volume.shift(2))
            & (volume.shift(2) < volume.shift(3))
            & (close.shift(1) >= floor)
            & (close.shift(2) >= floor)
        )
        echo = (
            spike.shift(3, fill_value=False)
            & digestion
            & (close > close.shift(3))
            & (close > data["open"])
            & (volume >= params.volume_ratio * (volume.shift(1) + volume.shift(2)) / 2)
        )
        return pd.DataFrame({
            "true_range": tr,
            "atr": atr,
            "volume_baseline": baseline,
            "digestion_floor": floor,
            "spike": spike.astype(int),
            "digestion": digestion.astype(int),
            "echo": echo.astype(int),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        active, age, next_entry = False, 0, 0
        high_water, stop = np.nan, -np.inf
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        echo = indicators["echo"].fillna(0).to_numpy(dtype=int)
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                age += 1
                if valid:
                    high_water = close[i] if np.isnan(high_water) else max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if not valid or close[i] <= stop or age >= 4:
                    active = False
                    next_entry = i + 3
            elif i >= max(self.warmup_bars(params) - 1, next_entry) and valid and echo[i] == 1:
                active, age = True, 0
                # The next bar is the fill bar; its close starts the in-trade high-water mark.
                high_water, stop = np.nan, -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=32)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 100.0,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=24)
    indicators = strategy.indicators(data, params)
    assert indicators.index.equals(data.index)
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].eq(0).all()
    indicators["echo"] = 0
    indicators.loc[index[23], "echo"] = 1
    indicators["atr"] = 1.0
    output = strategy.generate_signals(data, indicators, ctx, params).data
    assert output["signal"].tolist()[23:29] == [0, 1, 1, 1, 1, 0]
    data.loc[index[24:27], "close"] = [103.0, 103.0, 101.0]
    indicators.loc[index[25], "atr"] = 10.0
    output = strategy.generate_signals(data, indicators, ctx, params).data
    assert output["signal"].tolist()[23:28] == [0, 1, 1, 1, 0]
    assert output["size"].gt(0).all()
    assert set(output["signal"].unique()) <= {0, 1}


if __name__ == "__main__":
    _self_check()
