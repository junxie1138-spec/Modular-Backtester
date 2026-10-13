from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    lookback: int = 126
    atr_window: int = 14
    information_floor: float = 0.02
    edge_floor: float = 0.10
    min_samples: int = 8
    trail_atr: float = 3.0
    hold_bars: int = 18

    def __post_init__(self):
        if self.lookback < 36 or self.atr_window < 2 or self.min_samples < 3:
            raise ValueError("Insufficient estimation windows")
        if self.min_samples > self.lookback or not 15 <= self.hold_bars <= 20:
            raise ValueError("Invalid sample count or holding horizon")
        if not 0 < self.trail_atr < np.inf:
            raise ValueError("trail_atr must be finite and positive")
        if not 0 < self.information_floor < 1 or not 0 <= self.edge_floor < np.inf:
            raise ValueError("Invalid signal thresholds")


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local_1790965756"

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params):
        return int(max(params.lookback + 8, params.atr_window + 1, 21))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        a, b, c = out["return"].shift(2), out["return"].shift(1), out["return"]
        # Equal returns are ordered chronologically, giving deterministic motifs.
        motif = np.select(
            [(a <= b) & (b <= c), (a <= c) & (c < b),
             (b < a) & (a <= c), (b <= c) & (c < a),
             (c < a) & (a <= b)],
            [0, 1, 2, 3, 4], default=5,
        )
        out["pattern"] = pd.Series(motif, index=data.index, dtype=float).where(
            a.notna() & b.notna() & c.notna()
        )
        out["old_pattern"] = out["pattern"].shift(5)
        out["week_return"] = close.pct_change(5, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["volatility"] = out["return"].rolling(20, min_periods=20).std(ddof=0)
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        valid = out["pattern"].notna() & out["old_pattern"].notna() & out["week_return"].notna()
        out["pair_count"] = valid.astype(float).rolling(params.lookback, min_periods=params.lookback).sum()
        complete = out["pair_count"].eq(params.lookback)
        out["forecast"] = np.nan
        out["sample_count"] = 0.0
        entropy = pd.Series(0.0, index=data.index)
        mutual_information = pd.Series(0.0, index=data.index)
        active_old = pd.Series(0.0, index=data.index)
        active_new = pd.Series(0.0, index=data.index)
        for j in range(6):
            weights = out["old_pattern"].eq(j).astype(float).where(valid)
            counts = weights.rolling(params.lookback, min_periods=params.lookback).sum()
            out[f"count_{j}"] = counts
            out[f"old_probability_{j}"] = counts / params.lookback
            out[f"new_probability_{j}"] = out["pattern"].eq(j).astype(float).where(valid).rolling(
                params.lookback, min_periods=params.lookback
            ).mean()
            out[f"conditional_return_{j}"] = (weights * out["week_return"]).rolling(
                params.lookback, min_periods=params.lookback
            ).sum() / counts.replace(0, np.nan)
            selected = out["pattern"].eq(j)
            out.loc[selected, "forecast"] = out.loc[selected, f"conditional_return_{j}"]
            out.loc[selected, "sample_count"] = counts.loc[selected]
            probability = out[f"new_probability_{j}"]
            entropy -= (probability * np.log(probability.clip(lower=1e-12))).fillna(0)
            active_old += out[f"old_probability_{j}"].gt(0).astype(float)
            active_new += probability.gt(0).astype(float)
        for j in range(6):
            for k in range(6):
                joint = (out["old_pattern"].eq(j) & out["pattern"].eq(k)).astype(float).where(valid).rolling(
                    params.lookback, min_periods=params.lookback
                ).mean()
                out[f"joint_{j}_{k}"] = joint
                independent = out[f"old_probability_{j}"] * out[f"new_probability_{k}"]
                ratio = joint / independent.replace(0, np.nan)
                mutual_information += (joint * np.log(ratio.clip(lower=1e-12))).fillna(0)
        out["pattern_entropy"] = entropy.where(complete)
        out["mutual_information"] = mutual_information.where(complete)
        out["information_bias"] = ((active_old - 1) * (active_new - 1) / (2 * params.lookback)).where(complete)
        out["information"] = (
            (out["mutual_information"] - out["information_bias"]).clip(lower=0)
            / out["pattern_entropy"].replace(0, np.nan)
        ).clip(0, 1)
        out["edge"] = out["forecast"] / (out["volatility"].replace(0, np.nan) * np.sqrt(5.0))
        out["strength"] = (
            out["edge"].clip(0, 1) * (out["information"] / 0.10).clip(0, 1)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.15, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = (
            indicators["information"].ge(params.information_floor)
            & indicators["edge"].ge(params.edge_floor)
            & indicators["sample_count"].ge(params.min_samples)
            & indicators["strength"].notna()
        ).to_numpy(dtype=bool)
        strength = indicators["strength"].fillna(0).to_numpy(dtype=float)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        trade_size = 0.15
        for i in range(len(data)):
            if not np.isfinite(close[i]) or close[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                active = False
                continue
            if active:
                age += 1
                # Start tracking at the first filled bar, after the entry decision.
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] <= stop or age >= params.hold_bars:
                    active = False
            elif ready[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                trade_size = float(0.15 + 0.80 * np.clip(strength[i], 0, 1))
            signals[i] = int(active)
            sizes[i] = trade_size if active else 0.15
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.15).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = OrdinalParams()
    index = pd.date_range("2020-01-01", periods=8)
    close = pd.Series([100, 101, 106, 108, 104, 101, 102, 103], index=index, dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.0})
    indicators = pd.DataFrame({"atr": [1, 1, 1, 1, 4, 1, 1, 1],
                               "information": [0.1] + [0.0] * 7,
                               "edge": 0.5, "sample_count": 20.0,
                               "strength": 0.5}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.55)
    assert result["size"].gt(0).all()
    indicators["information"] = 0.1
    indicators["atr"] = 100.0
    longer = pd.concat([data.iloc[:1]] * 24, ignore_index=True)
    longer.index = pd.date_range("2020-01-01", periods=24)
    repeated = pd.concat([indicators.iloc[:1]] * 24, ignore_index=True)
    repeated.index = longer.index
    held = strategy.generate_signals(longer, repeated, ctx, params).data["signal"]
    assert held.iloc[0] == 0 and held.iloc[1:19].eq(1).all() and held.iloc[19] == 0
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=320)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(index)))), index=index)
    history = pd.DataFrame({"open": prices, "high": prices + 1, "low": prices - 1,
                            "close": prices, "volume": 1000.0})
    full = strategy.indicators(history, params)
    prefix = strategy.indicators(history.iloc[:250], params)
    pd.testing.assert_frame_equal(full.iloc[:250], prefix)
    assert full["information"].iloc[:strategy.warmup_bars(params) - 1].isna().all()
    flat = history.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    flat_result = strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data
    assert flat_result["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
