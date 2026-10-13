from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    fast: int = 5
    slow: int = 20
    block: int = 5
    history: int = 252
    neighbors: int = 7
    radius: float = 1.0
    edge: float = 0.005
    profit_target: float = 0.035
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791340043"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def _feature_bars(self, params):
        return max(params.fast, params.slow + 1) + 3 * params.block

    def warmup_bars(self, params):
        return int(params.history + params.hold_bars + 2 * self._feature_bars(params))

    def indicators(self, data, params):
        p = params
        if not (1 <= p.fast < p.slow and p.block >= 1 and p.neighbors >= 3
                and p.hold_bars >= 1 and p.history >= p.neighbors * p.hold_bars
                and np.isfinite([p.radius, p.edge, p.profit_target]).all()
                and p.radius > 0 and p.edge > 0 and p.profit_target > 0):
            raise ValueError("Invalid analog parameters")
        close = data["close"].astype(float).where(data["close"] > 0)
        opening = data["open"].astype(float).where(data["open"] > 0)
        out = pd.DataFrame(index=data.index)
        out["log_return"] = np.log(close).diff()
        out["fast_slope"] = out["log_return"].rolling(p.fast, min_periods=p.fast).mean()
        out["predicted_slope"] = out["log_return"].rolling(p.slow, min_periods=p.slow).mean().shift(1)
        out["volatility"] = out["log_return"].rolling(p.slow, min_periods=p.slow).std(ddof=0)
        out["innovation"] = (out["fast_slope"] - out["predicted_slope"]) / out["volatility"].where(out["volatility"] > 1e-12)
        out["shape_now"] = out["innovation"].rolling(p.block, min_periods=p.block).mean()
        out["shape_middle"] = out["shape_now"].shift(p.block)
        out["shape_old"] = out["shape_now"].shift(2 * p.block)
        # At row t, this label belongs to the decision at t - hold_bars.
        out["matured_return"] = close / opening.shift(p.hold_bars - 1) - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        shape = out[["shape_old", "shape_middle", "shape_now"]].to_numpy()
        labels = out["matured_return"].to_numpy()
        lower = np.full(len(data), np.nan)
        upper = np.full(len(data), np.nan)
        embargo = self._feature_bars(p)
        # ponytail: bounded O(bars * history) search; index neighbors if history grows substantially.
        for i in range(self.warmup_bars(p), len(data)):
            if not np.isfinite(shape[i]).all():
                continue
            end = i - p.hold_bars - embargo
            origins = np.arange(max(0, end - p.history + 1), end + 1)
            outcomes = labels[origins + p.hold_bars]
            distance = np.sqrt(np.mean((shape[origins] - shape[i]) ** 2, axis=1))
            valid = np.isfinite(distance) & np.isfinite(outcomes) & (distance <= p.radius)
            candidates = np.flatnonzero(valid)
            order = candidates[np.argsort(distance[candidates], kind="stable")]
            selected = []
            for j in order:
                if all(abs(int(origins[j]) - int(origins[k])) >= p.hold_bars for k in selected):
                    selected.append(int(j))
                    if len(selected) == p.neighbors:
                        break
            if len(selected) == p.neighbors:
                lower[i], upper[i] = np.quantile(outcomes[selected], [0.25, 0.75])
        out["analog_lower"] = lower
        out["analog_upper"] = upper
        out["candidate"] = np.select(
            [(out["analog_lower"] > p.edge) & (out["shape_now"] > 0),
             (out["analog_upper"] < -p.edge) & (out["shape_now"] < 0)],
            [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        candidate = indicators["candidate"].fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy()
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opening[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                else:
                    gain = position * (closing[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                        position = 0
            elif confirmed[i] != 0 and np.isfinite(closing[i]) and closing[i] > 0:
                position = int(confirmed[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = AnalogParams(hold_bars=3)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({c: 100.0 for c in ["open", "high", "low", "close", "volume"]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in [1, -1]:
        indicators = pd.DataFrame({"candidate": direction}, index=index)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, direction, 0]
        target_data = data.copy()
        target_data.iloc[2, target_data.columns.get_loc("close")] = 100 * (1 + direction * 0.04)
        result = strategy.generate_signals(target_data, indicators, ctx, params).data
        assert result["signal"].iloc[2] == direction and result["signal"].iloc[3] == 0
        assert result["size"].gt(0).all()
    params = AnalogParams()
    rng = np.random.default_rng(17)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 600)))
    sample = pd.DataFrame({"open": prices, "high": prices * 1.01, "low": prices * 0.99,
                           "close": prices, "volume": 1000000.0},
                          index=pd.date_range("2020-01-01", periods=600))
    full = strategy.indicators(sample, params)
    prefix = strategy.indicators(sample.iloc[:480], params)
    assert np.allclose(full.iloc[:480].to_numpy(), prefix.to_numpy(), equal_nan=True)
    flat = strategy.indicators(sample.assign(open=100.0, close=100.0), params)
    assert flat["candidate"].eq(0).all()


if __name__ == "__main__":
    _self_check()
