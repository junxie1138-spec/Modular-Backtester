from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    rank_window: int = 126
    analog_window: int = 252
    neighbors: int = 8
    max_distance: float = 0.20
    min_win_rate: float = 0.625
    min_median_return: float = 0.01
    profit_target: float = 0.04
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local_1791309429"

    @classmethod
    def params_type(cls):
        return ShapeParams

    @staticmethod
    def warmup_bars(params: ShapeParams) -> int:
        return int(params.rank_window + params.analog_window + params.max_hold + 14)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        if min(params.rank_window, params.analog_window, params.neighbors, params.max_hold) < 1:
            raise ValueError("Lookbacks, neighbor count and holding period must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        width = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["range_fraction"] = width / close
        out["close_location"] = (close - data["low"]) / width
        out["late_range"] = out["range_fraction"].rolling(3, min_periods=3).mean()
        out["early_range"] = out["late_range"].shift(3)
        out["late_volume"] = data["volume"].rolling(3, min_periods=3).mean()
        out["early_volume"] = out["late_volume"].shift(3)
        out["late_location"] = out["close_location"].rolling(3, min_periods=3).mean()
        out["early_location"] = out["late_location"].shift(3)
        out["range_shape"] = out["late_range"] / out["early_range"].where(out["early_range"] > 0)
        out["volume_shape"] = out["late_volume"] / out["early_volume"].where(out["early_volume"] > 0)
        out["location_shape"] = out["late_location"] - out["early_location"]
        rank_columns = []
        for name in ("range_shape", "volume_shape", "location_shape"):
            column = name + "_rank"
            out[column] = out[name].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
            rank_columns.append(column)
        features = out[rank_columns].to_numpy(dtype=float)
        prices = close.to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        win_rate = np.full(len(data), np.nan)
        median_return = np.full(len(data), np.nan)
        distance = np.full(len(data), np.nan)
        # ponytail: O(bars * analog_window); batch distances if the archive grows.
        for i in range(self.warmup_bars(params) - 2, len(data)):
            if not np.isfinite(features[i]).all():
                continue
            end = i - params.max_hold - 6
            candidates = np.arange(max(0, end - params.analog_window + 1), end + 1, 6)
            valid = np.isfinite(features[candidates]).all(axis=1)
            valid &= np.isfinite(opens[candidates + 1]) & (opens[candidates + 1] > 0)
            valid &= np.isfinite(prices[candidates + params.max_hold])
            candidates = candidates[valid]
            if len(candidates) < params.neighbors:
                continue
            distances = np.abs(features[candidates] - features[i]).mean(axis=1)
            selected = np.argsort(distances, kind="stable")[:params.neighbors]
            analogs = candidates[selected]
            outcomes = prices[analogs + params.max_hold] / opens[analogs + 1] - 1.0
            win_rate[i] = np.mean(outcomes > 0.0)
            median_return[i] = np.median(outcomes)
            distance[i] = np.mean(distances[selected])
        out["analog_win_rate"] = win_rate
        out["analog_median_return"] = median_return
        out["analog_distance"] = distance
        out["entry_ok"] = (
            (out["analog_win_rate"] >= params.min_win_rate)
            & (out["analog_median_return"] >= params.min_median_return)
            & (out["analog_distance"] <= params.max_distance)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        eligible = indicators["entry_ok"].fillna(False).astype(bool)
        confirmed = (eligible & eligible.shift(1, fill_value=False)).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_price = 0.0
        entry_bar = -1
        for i in range(len(data)):
            if pending:
                entry_price = opens[i]
                entry_bar = i
                active = np.isfinite(entry_price) and entry_price > 0
                pending = False
            if active:
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.max_hold
                if target_hit or time_up:
                    active = False
                else:
                    signal[i] = 1
            elif confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0:
                signal[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = ShapeParams()
    index = pd.date_range("2020-01-01", periods=25)
    data = pd.DataFrame({name: 100.0 for name in ("open", "high", "low", "close", "volume")}, index=index)
    indicators = pd.DataFrame({"entry_ok": [True, True] + [False] * 23}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0] + [1] * 18 + [0] * 5
    data.loc[index[2], "close"] = 105.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1] + [0] * 22
    n = 520
    index = pd.date_range("2020-01-01", periods=n)
    close = 100.0 * np.exp(0.0005 * np.arange(n) + 0.015 * np.sin(np.arange(n) / 7.0))
    data = pd.DataFrame({"open": close * 0.999, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": 1e6 * (1.1 + 0.1 * np.cos(np.arange(n) / 5.0))}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:480], params)
    pd.testing.assert_frame_equal(full.iloc[:480], prefix)
    assert set(strategy.generate_signals(data, full, None, params).data["signal"].unique()) <= {0, 1}


if __name__ == "__main__":
    _self_check()
