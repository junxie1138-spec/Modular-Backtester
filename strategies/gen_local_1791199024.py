from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalAccelerationParams:
    seasonal_weeks: int = 26
    volatility_bars: int = 20
    entry_strength: float = 0.12
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SeasonalAccelerationParams]):
    strategy_id = "gen_local_1791199024"

    @classmethod
    def params_type(cls):
        return SeasonalAccelerationParams

    def warmup_bars(self, params):
        return int(7 * (params.seasonal_weeks + 2) + params.volatility_bars + 2)

    def indicators(self, data, params):
        if params.seasonal_weeks < 8 or params.volatility_bars < 3:
            raise ValueError("Insufficient seasonal or volatility lookback")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["acceleration"] = out["return"].diff()
        out["volatility"] = out["return"].rolling(params.volatility_bars).std().replace(0.0, np.nan)
        out["acceleration_z"] = out["acceleration"] / out["volatility"]
        out["weekday"] = data.index.dayofweek
        out["matured_weekday"] = out["weekday"].shift(1)
        out["matured_acceleration"] = out["acceleration_z"].shift(1)
        out["matured_response"] = (
            data["close"] / data["open"].replace(0.0, np.nan) - 1.0
        ) / out["volatility"].shift(1)
        for name in ("seasonal_x", "seasonal_y", "seasonal_variance", "seasonal_covariance"):
            out[name] = np.nan
        # Each observation pairs yesterday's acceleration with today's intraday return.
        for weekday in range(5):
            history = out.loc[out["matured_weekday"].eq(weekday)]
            x = history["matured_acceleration"]
            y = history["matured_response"]
            xroll = x.rolling(params.seasonal_weeks)
            stats = pd.DataFrame({
                "seasonal_x": xroll.mean(),
                "seasonal_y": y.rolling(params.seasonal_weeks).mean(),
                "seasonal_variance": xroll.var(),
                "seasonal_covariance": xroll.cov(y),
            }).reindex(data.index).ffill()
            mask = out["weekday"].eq(weekday)
            out.loc[mask, stats.columns] = stats.loc[mask]
        out["seasonal_beta"] = (
            out["seasonal_covariance"] / out["seasonal_variance"].replace(0.0, np.nan)
        ).clip(-2.0, 2.0)
        out["strength"] = (
            out["seasonal_y"]
            + out["seasonal_beta"] * (out["acceleration_z"] - out["seasonal_x"])
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        if params.entry_strength <= 0 or params.profit_target <= 0 or params.max_hold_bars not in (1, 2):
            raise ValueError("Thresholds must be positive and holding period must be one or two bars")
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 0.2
        strength = indicators["strength"].to_numpy()
        acceleration = indicators["acceleration_z"].to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        direction, entry_bar, entry_price, entry_size = 0, -1, np.nan, 0.2
        raw_signal = np.zeros(len(data), dtype=int)
        raw_size = np.full(len(data), 0.2)
        for i in range(len(data)):
            if direction:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    direction = 0
                    continue
                gain = direction * (closes[i] / entry_price - 1.0)
                age = i - entry_bar + 1
                # Close-observed exits become market exits on the next bar.
                if gain >= params.profit_target or age >= params.max_hold_bars:
                    direction = 0
                    continue
            elif (
                np.isfinite(strength[i])
                and np.isfinite(acceleration[i])
                and abs(acceleration[i]) >= 0.75
                and abs(strength[i]) >= params.entry_strength
            ):
                direction = 1 if strength[i] > 0 else -1
                entry_bar = i + 1
                entry_price = np.nan
                entry_size = float(np.clip(0.2 + 0.6 * abs(strength[i]), 0.2, 1.0))
            raw_signal[i] = direction
            raw_size[i] = entry_size if direction else 0.2
        df["signal"] = raw_signal
        df["size"] = raw_size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SeasonalAccelerationParams(profit_target=0.005)
    index = pd.bdate_range("2020-01-01", periods=8)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        data["close"] = [100.0, 100.0 + direction, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        indicators = pd.DataFrame({"strength": float(direction), "acceleration_z": float(direction)}, index=index)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == [0, direction, 0, direction, direction, 0, direction, direction]
        assert result["size"].gt(0).all()
        weaker = indicators.copy()
        weaker["strength"] *= 0.5
        smaller = strategy.generate_signals(data, weaker, ctx, params).data
        assert smaller["size"].iloc[1] < result["size"].iloc[1]


if __name__ == "__main__":
    _self_check()
