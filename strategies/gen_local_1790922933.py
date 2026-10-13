from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_bars: int = 5
    history_bars: int = 504
    neighbors: int = 8
    holding_bars: int = 4
    profit_target: float = 0.018
    min_forecast: float = 0.004
    min_hit_rate: float = 0.60
    max_distance: float = 0.90
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790922933"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.history_bars + params.shape_bars + params.holding_bars + 1)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        if not (p.shape_bars >= 4 and p.history_bars >= p.shape_bars + p.holding_bars
                and p.neighbors >= 1 and 3 <= p.holding_bars <= 5
                and p.profit_target > 0 and p.min_forecast > 0
                and 0 < p.min_hit_rate <= 1 and p.max_distance > 0
                and 0 < p.min_size <= p.max_size <= 1):
            raise ValueError("Invalid analog parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        columns = []
        for lag in range(p.shape_bars - 1, -1, -1):
            name = f"return_lag_{lag}"
            out[name] = out["return"].shift(lag)
            columns.append(name)
        matrix = out[columns].to_numpy(dtype=float)
        out["rms"] = np.sqrt(out[columns].pow(2).mean(axis=1, skipna=False))
        rms = out["rms"].to_numpy(dtype=float)
        shape = matrix / np.where(rms > 1e-12, rms, np.nan)[:, None]
        for k in range(p.shape_bars):
            out[f"shape_{k}"] = shape[:, k]
        out["realized_horizon_return"] = data["close"].pct_change(
            p.holding_bars, fill_method=None
        ).replace([np.inf, -np.inf], np.nan)
        realized = out["realized_horizon_return"].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        hit_rate = np.full(len(data), np.nan)
        distance = np.full(len(data), np.nan)
        # ponytail: bounded O(bars * history / spacing) search; index only if history grows substantially.
        spacing = p.shape_bars + p.holding_bars
        for i in range(self.warmup_bars(p) - 1, len(data)):
            if not np.isfinite(shape[i]).all():
                continue
            # Every archived outcome ends before the current shape begins.
            ends = np.arange(i - p.history_bars, i - spacing + 1, spacing)
            valid = (np.isfinite(shape[ends]).all(axis=1)
                     & np.isfinite(realized[ends + p.holding_bars]))
            ends = ends[valid]
            if len(ends) < p.neighbors:
                continue
            d = np.mean((shape[ends] - shape[i]) ** 2, axis=1)
            d += np.log(rms[ends] / rms[i]) ** 2
            selected = np.argsort(d)[:p.neighbors]
            weights = 1.0 / (0.05 + d[selected])
            outcomes = realized[ends[selected] + p.holding_bars]
            forecast[i] = np.average(outcomes, weights=weights)
            hit_rate[i] = np.average(outcomes > 0, weights=weights)
            distance[i] = np.average(d[selected], weights=weights)
        out["forecast"] = forecast
        out["hit_rate"] = hit_rate
        out["distance"] = distance
        out["shock_recovery"] = (
            (out[columns[0]] < -out["rms"])
            & (out["return"] > 0)
            & (out[columns[-2:]].sum(axis=1, min_count=2)
               > out[columns[:2]].sum(axis=1, min_count=2))
        ).fillna(False)
        out["entry"] = (
            out["shock_recovery"]
            & (out["forecast"] >= p.min_forecast)
            & (out["hit_rate"] >= p.min_hit_rate)
            & (out["distance"] <= p.max_distance)
        ).fillna(False)
        out["strength"] = (
            (out["forecast"] / p.profit_target).clip(0, 1)
            * out["hit_rate"].clip(0, 1)
            * (1 - out["distance"] / p.max_distance).clip(0, 1)
        ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams,
    ) -> SignalFrame:
        p = params
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].fillna(0).clip(0, 1).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), p.min_size, dtype=float)
        entry_bar = -1
        entry_price = np.nan
        held_size = p.min_size
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # This open is observed only after yesterday's entry decision.
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    entry_bar = -1
                    continue
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + p.profit_target)
                time_up = i - entry_bar + 1 >= p.holding_bars
                if target_hit or time_up:
                    entry_bar = -1
                    continue
                signals[i] = 1
                sizes[i] = held_size
            elif entries[i]:
                entry_bar = i + 1
                held_size = p.min_size + (p.max_size - p.min_size) * strength[i]
                signals[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(p.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = AnalogParams()
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[2], "close"] = 103.0
    features = pd.DataFrame({"entry": False, "strength": 0.7}, index=index)
    features.loc[index[[0, 4]], "entry"] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
    assert np.isclose(result["size"].iloc[1], params.min_size + 0.7 * (params.max_size - params.min_size))
