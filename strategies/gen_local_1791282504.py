from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilitySurpriseParams:
    lookback: int = 20
    surprise_z: float = 1.5
    min_body_share: float = 0.65
    max_gap_share: float = 0.25
    stop_atr: float = 1.1


class GeneratedStrategy(BaseStrategy[VolatilitySurpriseParams]):
    strategy_id = "gen_local_1791282504"

    @classmethod
    def params_type(cls):
        return VolatilitySurpriseParams

    def warmup_bars(self, params: VolatilitySurpriseParams) -> int:
        return int(params.lookback) + 2

    def indicators(self, data: pd.DataFrame, params: VolatilitySurpriseParams) -> pd.DataFrame:
        if params.lookback < 2:
            raise ValueError("lookback must be at least two")
        if not np.isfinite(params.surprise_z) or params.surprise_z <= 0:
            raise ValueError("surprise_z must be positive and finite")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        if not 0 <= params.min_body_share <= 1 or not 0 <= params.max_gap_share <= 1:
            raise ValueError("body and gap shares must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        window = out["true_range"].rolling(params.lookback, min_periods=params.lookback)
        out["atr"] = window.mean()
        out["prior_mean"] = out["atr"].shift(1)
        out["prior_std"] = window.std(ddof=0).shift(1)
        out["surprise_scale"] = out["prior_std"].clip(lower=0.05 * out["prior_mean"])
        out["surprise_scale"] = out["surprise_scale"].where(out["surprise_scale"] > 0)
        out["surprise"] = (out["true_range"] - out["prior_mean"]) / out["surprise_scale"]
        positive_range = out["true_range"].where(out["true_range"] > 0)
        out["body_share"] = (data["close"] - data["open"]) / positive_range
        out["gap_share"] = (data["open"] - out["previous_close"]).abs() / positive_range
        out = out.replace([np.inf, -np.inf], np.nan)
        out["strength"] = ((out["surprise"] - params.surprise_z) / 2.0).clip(0.0, 1.0).fillna(0.0)
        out["entry"] = (
            (out["surprise"] >= params.surprise_z)
            & (out["body_share"] >= params.min_body_share)
            & (out["gap_share"] <= params.max_gap_share)
            & (data["close"] > out["previous_high"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VolatilitySurpriseParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        strength = indicators["strength"].fillna(0.0).to_numpy(dtype=float)
        entry_bar = -1
        stop = np.nan
        stop_distance = 0.0
        position_size = 1.0
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Anchor to the entry opening price; freeze the decision-bar ATR distance.
                    stop = opens[i] - stop_distance
                if (not np.isfinite(stop) or not np.isfinite(closes[i])
                        or closes[i] < stop or i - entry_bar + 1 >= 2):
                    entry_bar = -1
                else:
                    signals[i] = 1
                    sizes[i] = position_size
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                entry_bar = i + 1
                stop_distance = params.stop_atr * atr[i]
                position_size = 0.35 + 0.65 * float(np.clip(strength[i], 0.0, 1.0))
                signals[i] = 1
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = VolatilitySurpriseParams()
    index = pd.date_range("2020-01-01", periods=7)
    close = np.array([100.0, 97.0, 100.0, 100.0, 110.0, 99.0, 100.0])
    data = pd.DataFrame({
        "open": 100.0,
        "high": np.maximum(close, 100.0) + 1.0,
        "low": np.minimum(close, 100.0) - 1.0,
        "close": close,
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, True, False, False, False],
        "atr": 2.0,
        "strength": [0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=22)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 1, 1, 0]
    assert np.isclose(result["size"].iloc[1], 0.48)
    assert np.allclose(result["size"].iloc[4:6], 0.87)
    for end in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:end], indicators.iloc[:end], ctx, params).data
        pd.testing.assert_frame_equal(prefix, result.iloc[:end])
