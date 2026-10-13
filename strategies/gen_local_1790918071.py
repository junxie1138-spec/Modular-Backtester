from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    innovation_threshold: float = 0.10
    trail_atr: float = 1.6


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local_1790918071"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return 60

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["floor"] = close.rolling(20, min_periods=20).min().shift(1)
        out["ceiling"] = close.rolling(20, min_periods=20).max().shift(1)
        width = (out["ceiling"] - out["floor"]).replace(0.0, np.nan)
        out["relative_position"] = (close - out["floor"]) / width
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        positions = out["relative_position"].to_numpy(dtype=float)
        predictions = np.full(len(data), np.nan)
        innovations = np.full(len(data), np.nan)
        levels = np.full(len(data), np.nan)
        velocities = np.full(len(data), np.nan)
        level = np.nan
        velocity = 0.0
        for i, position in enumerate(positions):
            if not np.isfinite(position):
                level = np.nan
                velocity = 0.0
                continue
            if not np.isfinite(level):
                level = position
                prediction = position
                innovation = 0.0
            else:
                prediction = level + velocity
                innovation = position - prediction
                level = prediction + 0.35 * innovation
                velocity += 0.08 * innovation
            predictions[i] = prediction
            innovations[i] = innovation
            levels[i] = level
            velocities[i] = velocity
        out["prediction"] = predictions
        out["innovation"] = innovations
        out["latent_position"] = levels
        out["latent_velocity"] = velocities
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold <= 0:
            raise ValueError("innovation_threshold must be positive and finite")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        positions = indicators["relative_position"].to_numpy(dtype=float)
        innovations = indicators["innovation"].to_numpy(dtype=float)
        velocities = indicators["latent_velocity"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        high_water = low_water = stop = np.nan
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = (
                i >= self.warmup_bars(params)
                and np.isfinite(close)
                and np.isfinite(atr)
                and atr > 0
                and np.isfinite(positions[i])
                and np.isfinite(innovations[i])
                and np.isfinite(velocities[i])
            )
            if not valid:
                side = 0
                continue
            if side:
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                if side == 1:
                    stop = max(stop, high_water - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    stop = min(stop, low_water + params.trail_atr * atr)
                    stopped = close >= stop
                if stopped or i - entry_bar >= 4:
                    side = 0
                signals[i] = side
                continue
            if 0.20 <= positions[i] <= 0.80:
                if innovations[i] >= params.innovation_threshold and velocities[i] > 0:
                    side = 1
                elif innovations[i] <= -params.innovation_threshold and velocities[i] < 0:
                    side = -1
            if side:
                entry_bar = i
                high_water = low_water = close
                stop = close - side * params.trail_atr * atr
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    t = np.arange(160, dtype=float)
    close = 100 + 0.03 * t + 2 * np.sin(t / 4)
    sample = pd.DataFrame({
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": 1000000.0,
    }, index=pd.date_range("2020-01-01", periods=len(t)))
    strategy = GeneratedStrategy()
    params = ShockwaveParams(innovation_threshold=0.03)
    derived = strategy.indicators(sample, params)
    result = strategy.generate_signals(sample, derived, None, params).data
    prefix = sample.iloc[:120]
    prefix_result = strategy.generate_signals(
        prefix, strategy.indicators(prefix, params), None, params
    ).data
    pd.testing.assert_frame_equal(result.iloc[:120], prefix_result)
    assert result["signal"].isin([-1, 0, 1]).all()
    assert (result["size"] > 0).all()
    assert (result["signal"].iloc[:61] == 0).all()
    assert (result["signal"] != 0).any()
    runs = result["signal"].ne(result["signal"].shift()).cumsum()
    assert result.loc[result["signal"] != 0].groupby(runs).size().max() <= 4
