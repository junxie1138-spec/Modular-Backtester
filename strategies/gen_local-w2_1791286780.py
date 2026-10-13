from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    profit_target: float = 0.04
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791286780"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")
        return 540

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["daily_return"] = close.pct_change(fill_method=None)
        out["volatility"] = out["daily_return"].rolling(20, min_periods=20).std()
        out["roc"] = close.pct_change(5, fill_method=None)
        out["acceleration"] = out["roc"].diff(5)
        out["scaled_acceleration"] = out["acceleration"] / (out["volatility"].replace(0, np.nan) * np.sqrt(10.0))
        offsets = (0, 3, 6, 9)
        for offset in offsets:
            out[f"shape_{offset}"] = out["scaled_acceleration"].shift(offset)
        lags = tuple(range(42, 505, 21))
        distance_columns = []
        outcome_columns = []
        for lag in lags:
            distance = pd.Series(0.0, index=data.index)
            for offset in offsets:
                historical = f"historical_{lag}_{offset}"
                out[historical] = out[f"shape_{offset}"].shift(lag)
                distance = distance + (out[f"shape_{offset}"] - out[historical]).pow(2)
            distance_column = f"distance_{lag}"
            outcome_column = f"outcome_{lag}"
            out[distance_column] = np.sqrt(distance / len(offsets))
            # Every eighteen-session outcome finished at least 24 bars ago.
            out[outcome_column] = close.shift(lag - 18) / close.shift(lag) - 1.0
            distance_columns.append(distance_column)
            outcome_columns.append(outcome_column)
        distances = out[distance_columns].to_numpy(dtype=float)
        outcomes = out[outcome_columns].to_numpy(dtype=float)
        valid = np.isfinite(distances) & np.isfinite(outcomes)
        distances = np.where(valid, distances, np.inf)
        nearest = np.argsort(distances, axis=1, kind="stable")[:, :5]
        selected_distances = np.take_along_axis(distances, nearest, axis=1)
        selected_outcomes = np.take_along_axis(outcomes, nearest, axis=1)
        complete = np.isfinite(selected_distances).all(axis=1)
        safe_outcomes = np.where(np.isfinite(selected_outcomes), selected_outcomes, 0.0)
        out["forecast"] = np.where(complete, safe_outcomes.mean(axis=1), np.nan)
        out["analog_distance"] = np.where(complete, np.where(np.isfinite(selected_distances), selected_distances, 0.0).mean(axis=1), np.nan)
        out["positive_votes"] = np.where(complete, (safe_outcomes > 0).sum(axis=1), 0)
        out["negative_votes"] = np.where(complete, (safe_outcomes < 0).sum(axis=1), 0)
        quality = complete & (out["analog_distance"] <= 1.25)
        long_entry = quality & (out["forecast"] > 0.01) & (out["positive_votes"] >= 4) & (out["acceleration"] > 0)
        short_entry = quality & (out["forecast"] < -0.01) & (out["negative_votes"] >= 4) & (out["acceleration"] < 0)
        out["entry_direction"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        out = out.replace([np.inf, -np.inf], np.nan)
        out.loc[: , "entry_direction"] = out["entry_direction"].fillna(0).astype(int)
        out.iloc[:self.warmup_bars(params), out.columns.get_loc("entry_direction")] = 0
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        self.warmup_bars(params)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if position != 0:
                # An earlier close decision fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    age = 0
                age += 1
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                gain = position * (closes[i] / entry_price - 1.0) if not invalid_entry and np.isfinite(closes[i]) else np.nan
                if invalid_entry or age >= params.holding_bars or (np.isfinite(gain) and gain >= params.profit_target):
                    position = 0
                    entry_price = np.nan
                    age = 0
                    # Exit decisions fill at the following open; no same-close reentry.
                    decisions[i] = 0
                    continue
            elif entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_price = np.nan
                age = 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    sample = pd.DataFrame({"open": 100.0, "high": 105.0, "low": 95.0, "close": [100.0, 101.0, 104.0, 100.0, 100.0, 99.0, 96.0, 100.0], "volume": 1000.0}, index=index)
    evidence = pd.DataFrame({"entry_direction": [1, 0, 0, 0, -1, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(sample, evidence, None, AnalogParams(profit_target=0.03, holding_bars=18))
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    sample["close"] = 100.0
    evidence["entry_direction"] = [1, 0, 0, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(sample, evidence, None, AnalogParams(holding_bars=3))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
