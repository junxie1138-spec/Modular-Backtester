from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    trend_window: int = 6
    volatility_window: int = 20
    atr_window: int = 14
    archive_bars: int = 126
    bandwidth: float = 1.0
    forecast_threshold: float = 0.15
    stop_atr: float = 1.25
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791344293"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return max(params.volatility_window + 1, params.trend_window + 2,
                   params.atr_window + 1) + params.archive_bars + 3

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if (min(params.trend_window, params.volatility_window, params.atr_window) < 2
                or params.archive_bars < 10 or params.bandwidth <= 0
                or params.forecast_threshold <= 0 or params.stop_atr <= 0
                or params.hold_days not in (1, 2)):
            raise ValueError("Invalid analog parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["volatility"] = out["return"].rolling(params.volatility_window).std().replace(0, np.nan)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean().replace(0, np.nan)
        out["latent_slope"] = out["return"].rolling(params.trend_window).mean()
        out["innovation"] = out["latent_slope"].diff() * params.trend_window / out["volatility"]
        out["innovation_lag1"] = out["innovation"].shift(1)
        out["innovation_lag2"] = out["innovation"].shift(2)
        out["slope_state"] = out["latent_slope"] / out["volatility"]
        features = out[["innovation", "innovation_lag1", "innovation_lag2", "slope_state"]]
        numerator = pd.Series(0.0, index=data.index)
        support = pd.Series(0.0, index=data.index)
        # ponytail: O(bars * archive_bars); use a causal neighbor index for much larger archives.
        for lag in range(7, params.archive_bars + 1):
            distance = (features - features.shift(lag)).pow(2).sum(axis=1, min_count=4) / 4.0
            outcome = (out["log_close"].shift(lag - 2) - out["log_close"].shift(lag)) / out["volatility"].shift(lag)
            weight = np.exp(-distance / (2.0 * params.bandwidth ** 2))
            weight = weight.where(np.isfinite(outcome), 0.0).fillna(0.0)
            numerator += weight * outcome.fillna(0.0)
            support += weight
        out["analog_support"] = support
        out["forecast"] = (numerator / support.replace(0, np.nan)).where(support >= 3.0)
        ready = features.notna().all(axis=1) & out["atr"].notna()
        ready &= np.arange(len(data)) >= self.warmup_bars(params) - 1
        out["candidate"] = np.select([
            ready & (out["forecast"] > params.forecast_threshold),
            ready & (out["forecast"] < -params.forecast_threshold)
        ], [1, -1], default=0).astype(int)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].eq(out["candidate"].shift(1)), 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high_water, low_water = -np.inf, np.inf
        lower_stop, upper_stop = -np.inf, np.inf
        for i, (close, atr, confirmed) in enumerate(zip(
                data["close"].to_numpy(), indicators["atr"].to_numpy(),
                indicators["confirmed"].fillna(0).to_numpy())):
            valid = np.isfinite(close) and np.isfinite(atr) and close > 0 and atr > 0
            if position:
                age += 1
                if valid:
                    # First held bar is the bar after the entry decision.
                    high_water = max(high_water, close)
                    low_water = min(low_water, close)
                    lower_stop = max(lower_stop, high_water - params.stop_atr * atr)
                    upper_stop = min(upper_stop, low_water + params.stop_atr * atr)
                # The high-water drawdown exit applies to either direction;
                # shorts additionally have the mirrored adverse-price trail.
                if (not valid or close <= lower_stop
                        or (position == -1 and close >= upper_stop)
                        or age >= params.hold_days):
                    position = 0
            elif valid and confirmed in (-1, 1):
                position, age = int(confirmed), 0
                high_water, low_water = -np.inf, np.inf
                lower_stop, upper_stop = -np.inf, np.inf
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=220)
    close = 100.0 * np.exp(np.cumsum(0.003 * np.sin(np.arange(220) * 0.7)))
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0,
                         "close": close, "volume": 1000.0}, index=index)
    params = AnalogParams(archive_bars=30)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:170], params)
    pd.testing.assert_frame_equal(full.iloc[:170], prefix)
    expected = full["candidate"].where(full["candidate"].eq(full["candidate"].shift(1)), 0)
    assert full["confirmed"].equals(expected)
    fixture = data.iloc[:9].copy()
    fixture["close"] = [100, 100, 105, 103, 100, 100, 95, 97, 100]
    ind = pd.DataFrame({"atr": 1.0, "confirmed": [0, 1, 1, 1, 0, -1, -1, -1, 0]},
                       index=fixture.index)
    result = strategy.generate_signals(fixture, ind, None, AnalogParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, -1, -1, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
