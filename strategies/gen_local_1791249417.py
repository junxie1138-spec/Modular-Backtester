from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    season_bars: int = 252
    pressure_bars: int = 5
    atr_bars: int = 14
    hold_bars: int = 8
    season_threshold: float = 1.0
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791249417"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.season_bars, params.pressure_bars, params.atr_bars) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.season_bars, params.pressure_bars, params.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 5 <= params.hold_bars <= 10:
            raise ValueError("hold_bars must be between five and ten.")
        if params.season_threshold <= 0 or params.trail_atr <= 0:
            raise ValueError("Threshold and trail multiplier must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return"] = returns
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        weekday = data.index.dayofweek.to_numpy()
        forecast = pd.Series(0.0, index=data.index)
        uncertainty = pd.Series(0.0, index=data.index)
        for day in range(5):
            mask = pd.Series((weekday == day) & returns.notna().to_numpy(), index=data.index)
            count = mask.astype(float).rolling(params.season_bars).sum()
            total = returns.where(mask, 0.0).rolling(params.season_bars).sum()
            squares = returns.pow(2).where(mask, 0.0).rolling(params.season_bars).sum()
            mean = (total / count.replace(0, np.nan)).where(count >= 10)
            variance = ((squares - total.pow(2) / count.replace(0, np.nan))
                        / (count - 1).replace(0, np.nan)).clip(lower=0).where(count >= 10)
            out[f"weekday_count_{day}"] = count
            out[f"weekday_mean_{day}"] = mean
            out[f"weekday_variance_{day}"] = variance
            # ponytail: weekday weights omit holidays; add an exchange calendar if displacement matters.
            weight = sum(((weekday + step) % 5 == day).astype(float)
                         for step in range(1, params.hold_bars + 1))
            forecast = forecast + weight * mean
            uncertainty = uncertainty + weight ** 2 * variance / count.replace(0, np.nan)
        out["season_forecast"] = forecast
        out["season_se"] = np.sqrt(uncertainty)
        out["season_score"] = forecast / out["season_se"].replace(0, np.nan)
        volume = data["volume"].clip(lower=0)
        out["return_volume"] = returns * volume
        out["pressure"] = (out["return_volume"].rolling(params.pressure_bars).sum()
                           / volume.rolling(params.pressure_bars).sum().replace(0, np.nan))
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        score = indicators["season_score"]
        pressure = indicators["pressure"]
        long_entry = ((score >= params.season_threshold) & (pressure > 0)).fillna(False).to_numpy()
        short_entry = ((score <= -params.season_threshold) & (pressure < 0)).fillna(False).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = age = 0
        high_water, low_water, stop = -np.inf, np.inf, np.nan
        for i, (close, atr) in enumerate(zip(closes, atrs)):
            if not np.isfinite(close) or not np.isfinite(atr) or close <= 0 or atr <= 0:
                side = age = 0
                high_water, low_water, stop = -np.inf, np.inf, np.nan
                continue
            if side:
                # Entry fills on this bar following yesterday's decision.
                age += 1
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                if side == 1:
                    candidate = high_water - params.trail_atr * atr
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    stopped = close <= stop
                else:
                    # Short-side mirror: the low-water stop only ratchets downward.
                    candidate = low_water + params.trail_atr * atr
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    stopped = close >= stop
                if stopped or age >= params.hold_bars:
                    side = age = 0
                    high_water, low_water, stop = -np.inf, np.inf, np.nan
            elif long_entry[i] or short_entry[i]:
                side = 1 if long_entry[i] else -1
                age = 0
                high_water, low_water, stop = -np.inf, np.inf, np.nan
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = Params(trail_atr=2.0)
    index = pd.bdate_range("2020-01-01", periods=6)
    for side, closes in [(1, [100, 100, 110, 109, 107, 107]),
                         (-1, [100, 100, 90, 91, 93, 93])]:
        data = pd.DataFrame({"close": closes}, index=index)
        derived = pd.DataFrame({"season_score": side * 2.0, "pressure": float(side),
                                "atr": [1, 1, 1, 5, 5, 5]}, index=index)
        result = GeneratedStrategy.generate_signals(data, derived, None, params).data
        assert result["signal"].tolist() == [0, side, side, side, side, 0]
        assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
