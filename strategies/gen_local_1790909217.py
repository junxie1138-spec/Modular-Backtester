from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    peak_window: int = 40
    analog_window: int = 252
    neighbors: int = 9
    holding_days: int = 4
    atr_window: int = 14
    min_depth: float = 0.01
    forecast_threshold: float = 0.002
    migration_threshold: float = 0.10
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790909217"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(params.peak_window + 5 + params.analog_window + params.holding_days + 1,
                       params.atr_window + 1, 4))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if (params.peak_window < 2 or params.analog_window < params.neighbors
                or params.neighbors < 1 or params.holding_days not in (3, 4, 5)
                or params.atr_window < 1
                or not all(np.isfinite(x) and x > 0 for x in
                           (params.min_depth, params.forecast_threshold,
                            params.migration_threshold, params.stop_atr))):
            raise ValueError("Invalid analog parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["peak"] = close.rolling(params.peak_window).max()
        out["depth"] = 1.0 - close / out["peak"].where(out["peak"] > 0)
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out.loc[previous.isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["range_center"] = (data["high"] + data["low"]) / 2.0
        out["center_change"] = out["range_center"].diff()
        out["migration"] = out["center_change"].rolling(3).mean() / out["atr"].where(out["atr"] > 0)
        columns = []
        for lag in (0, 1, 3, 5):
            name = f"depth_lag_{lag}"
            out[name] = out["depth"].shift(lag)
            columns.append(name)
        shapes = out[columns].to_numpy(dtype=float)
        closes = close.to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        forecasts = np.full(len(data), np.nan)
        # ponytail: bounded daily neighbor scan; use an index only for much larger histories.
        for t in range(self.warmup_bars(params) - 1, len(data)):
            if not np.isfinite(shapes[t]).all():
                continue
            # Each analog's entire outcome ends strictly before the decision bar.
            end = t - params.holding_days
            candidates = np.arange(max(0, end - params.analog_window), end)
            entry = opens[candidates + 1]
            exit_close = closes[candidates + params.holding_days]
            valid = (np.isfinite(shapes[candidates]).all(axis=1)
                     & np.isfinite(entry) & (entry > 0)
                     & np.isfinite(exit_close) & (exit_close > 0))
            candidates = candidates[valid]
            if len(candidates) < params.neighbors:
                continue
            distances = np.square(shapes[candidates] - shapes[t]).sum(axis=1)
            nearest = candidates[np.argsort(distances, kind="stable")[:params.neighbors]]
            outcomes = closes[nearest + params.holding_days] / opens[nearest + 1] - 1.0
            forecasts[t] = float(np.median(outcomes))
        out["forecast"] = forecasts
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        ready = (indicators[["depth", "forecast", "migration", "atr"]].notna().all(axis=1)
                 & (indicators["depth"] >= params.min_depth) & (indicators["atr"] > 0))
        long_entry = (ready & (indicators["forecast"] > params.forecast_threshold)
                      & (indicators["migration"] > params.migration_threshold))
        short_entry = (ready & (indicators["forecast"] < -params.forecast_threshold)
                       & (indicators["migration"] < -params.migration_threshold))
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        fixed_distance = np.nan
        for t in range(len(data)):
            if position:
                if t == entry_bar:
                    entry_price = opens[t]
                valid_price = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[t])
                stopped = valid_price and (
                    closes[t] < entry_price - fixed_distance if position == 1
                    else closes[t] > entry_price + fixed_distance
                )
                if not valid_price or stopped or t - entry_bar + 1 >= params.holding_days:
                    position = 0
                    desired[t] = 0
                    continue
            elif entries[t] and t >= self.warmup_bars(params) - 1:
                position = int(entries[t])
                entry_bar = t + 1
                entry_price = np.nan
                fixed_distance = params.stop_atr * atr[t]
            desired[t] = position
        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = AnalogParams(peak_window=2, analog_window=9, neighbors=3)
    count = 100
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, count)))
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0,
                         "close": close, "volume": 1000.0},
                        index=pd.date_range("2020-01-01", periods=count))
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:80], params)
    assert np.allclose(full.iloc[:80].to_numpy(), prefix.to_numpy(), equal_nan=True)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    start = strategy.warmup_bars(params) - 1
    prices = data.copy()
    prices[["open", "close"]] = 100.0
    forced = pd.DataFrame({"depth": 0.03, "forecast": 0.01, "migration": 0.5,
                           "atr": 1.0}, index=data.index)
    prices.iloc[start + 2, prices.columns.get_loc("close")] = 104.0
    result = strategy.generate_signals(prices, forced, ctx, params).data
    assert result["signal"].iloc[start] == 0
    assert result["signal"].iloc[start + 1:start + 5].tolist() == [1, 1, 1, 1]
    assert result["signal"].iloc[start + 5] == 0
    prices.iloc[start + 3, prices.columns.get_loc("close")] = 98.0
    result = strategy.generate_signals(prices, forced, ctx, params).data
    assert result["signal"].iloc[start + 3] == 1 and result["signal"].iloc[start + 4] == 0
    prices["close"] = 100.0
    prices.iloc[start + 3, prices.columns.get_loc("close")] = 102.0
    forced["forecast"], forced["migration"] = -0.01, -0.5
    result = strategy.generate_signals(prices, forced, ctx, params).data
    assert result["signal"].iloc[start + 3] == -1 and result["signal"].iloc[start + 4] == 0
    assert result["size"].gt(0).all() and set(result["signal"]).issubset({-1, 0, 1})


if __name__ == "__main__":
    _self_check()
