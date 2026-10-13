from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    path_bars: int = 4
    tension_bars: int = 12
    atr_bars: int = 14
    min_tension: float = 1.0
    area_threshold: float = 0.12


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791331881"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(200, params.tension_bars, params.atr_bars + params.path_bars + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if params.path_bars < 2 or params.tension_bars < 2 or params.atr_bars < 1:
            raise ValueError("Lookbacks must be positive; path and tension require at least two bars.")
        if not np.isfinite(params.min_tension) or params.min_tension <= 0:
            raise ValueError("min_tension must be finite and positive.")
        if not np.isfinite(params.area_threshold) or params.area_threshold <= 0:
            raise ValueError("area_threshold must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["price_change"] = close.diff()
        out["increment"] = out["price_change"] / out["previous_atr"].where(out["previous_atr"] > 0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["recent_high"] = close.rolling(params.tension_bars, min_periods=params.tension_bars).max()
        out["recent_low"] = close.rolling(params.tension_bars, min_periods=params.tension_bars).min()
        scale = out["atr"].where(out["atr"] > 0)
        out["drawdown"] = (out["recent_high"] - close) / scale
        out["drawup"] = (close - out["recent_low"]) / scale
        # S(time, price) - S(price, time), with equal time increments.
        # Positive area means later increments exceed earlier increments.
        area = pd.Series(0.0, index=data.index)
        for j in range(params.path_bars):
            area = area + (2 * j - params.path_bars + 1) * out["increment"].shift(params.path_bars - 1 - j)
        out["area"] = area / float(params.path_bars ** 2)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        i = indicators
        valid = i[["ma200", "increment", "drawdown", "drawup", "area"]].notna().all(axis=1)
        positive = i["area"] > params.area_threshold
        negative = i["area"] < -params.area_threshold
        long_entry = (valid & positive & (data["close"] > i["ma200"]) & (i["drawdown"] >= params.min_tension) & (i["increment"] > 0)).to_numpy()
        short_entry = (valid & negative & (data["close"] < i["ma200"]) & (i["drawup"] >= params.min_tension) & (i["increment"] < 0)).to_numpy()
        positive = positive.fillna(False).to_numpy()
        negative = negative.fillna(False).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for bar in range(len(data)):
            # Only an opposite signature predicate exits an existing position.
            # Regime and displacement filters govern entry, not exit.
            if position == 1:
                if negative[bar]:
                    position = 0
            elif position == -1:
                if positive[bar]:
                    position = 0
            elif long_entry[bar]:
                position = 1
            elif short_entry[bar]:
                position = -1
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"ma200": [90, 90, 90, 110, 110, 90], "increment": [0, 1, 1, -1, -1, 1], "drawdown": 2.0, "drawup": 2.0, "area": [0, 0.3, 0.2, -0.3, -0.3, 0.3]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1]
    assert result.index.equals(index) and (result["size"] > 0).all()
    assert strategy.warmup_bars(params) >= 200
