from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalRegimeParams:
    pattern_window: int = 63
    complexity_threshold: float = 0.45
    price_window: int = 18
    entry_strength: float = 1.0
    profit_target: float = 0.045
    hold_bars: int = 18
    min_size: float = 0.15


class GeneratedStrategy(BaseStrategy[OrdinalRegimeParams]):
    strategy_id = "gen_local-w2_1791244349"

    @classmethod
    def params_type(cls):
        return OrdinalRegimeParams

    def warmup_bars(self, params):
        return int(max(params.pattern_window + 3, params.price_window + 1))

    def indicators(self, data, params):
        if params.pattern_window < 6 or params.price_window < 3:
            raise ValueError("Lookback windows are too short")
        if not 0 < params.complexity_threshold < 1:
            raise ValueError("complexity_threshold must be in (0, 1)")
        if params.entry_strength <= 0 or params.profit_target <= 0:
            raise ValueError("Entry strength and profit target must be positive")
        if params.hold_bars < 1 or not 0 < params.min_size <= 1:
            raise ValueError("Invalid holding period or minimum size")
        close = data["close"].where(data["close"] > 0)
        triples = pd.concat([close.shift(2), close.shift(1), close], axis=1)
        # Stable sorting resolves tied prices chronologically.
        order = np.argsort(triples.to_numpy(), axis=1, kind="stable")
        code = pd.Series(3 * order[:, 0] + order[:, 1], index=data.index)
        pattern = code.map({1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5})
        pattern = pattern.where(triples.notna().all(axis=1))
        previous = pattern.shift(1)
        pair = previous * 6 + pattern
        valid = pair.notna()
        marginal_entropy = pd.Series(0.0, index=data.index)
        joint_entropy = pd.Series(0.0, index=data.index)
        for k in range(6):
            probability = (previous == k).astype(float).where(valid).rolling(params.pattern_window).mean()
            marginal_entropy += -probability * np.log(probability.clip(lower=1e-12))
        for k in range(36):
            probability = (pair == k).astype(float).where(valid).rolling(params.pattern_window).mean()
            joint_entropy += -probability * np.log(probability.clip(lower=1e-12))
        entropy = (marginal_entropy / np.log(6.0)).clip(0, 1)
        conditional_entropy = ((joint_entropy - marginal_entropy) / np.log(6.0)).clip(0, 1)
        complexity = entropy * (1.0 - conditional_entropy)
        returns = close.pct_change(fill_method=None)
        volatility = returns.rolling(params.price_window).std(ddof=0).replace(0, np.nan)
        mean = close.rolling(params.price_window).mean()
        dispersion = close.rolling(params.price_window).std(ddof=0).replace(0, np.nan)
        stretch = (close - mean) / dispersion
        momentum = close.pct_change(params.price_window, fill_method=None) / (volatility * np.sqrt(params.price_window))
        cyclic = complexity >= params.complexity_threshold
        edge = momentum.where(~cyclic, -stretch)
        regime_strength = ((complexity - params.complexity_threshold).abs() / 0.20).clip(0, 1)
        conviction = ((edge.abs() / params.entry_strength - 1.0) / 2.0).clip(0, 1)
        size = params.min_size + (1.0 - params.min_size) * conviction * regime_strength
        return pd.DataFrame({
            "pattern": pattern,
            "transition": pair,
            "entropy": entropy,
            "conditional_entropy": conditional_entropy,
            "complexity": complexity,
            "returns": returns,
            "volatility": volatility,
            "mean": mean,
            "dispersion": dispersion,
            "stretch": stretch,
            "momentum": momentum,
            "edge": edge,
            "regime_strength": regime_strength,
            "conviction": conviction,
            "entry_size": size,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        count = len(data)
        signal = np.zeros(count, dtype=int)
        sizes = np.ones(count, dtype=float)
        edges = indicators["edge"].to_numpy()
        complexity = indicators["complexity"].to_numpy()
        entry_sizes = indicators["entry_size"].to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        direction = 0
        entry_bar = -1
        entry_price = np.nan
        active_size = 1.0
        for i in range(count):
            if direction:
                if entry_bar < 0:
                    # The preceding close's decision fills at this bar's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        direction = 0
                        continue
                    entry_bar = i
                    entry_price = opens[i]
                gain = direction * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                    direction = 0
                    entry_bar = -1
                    continue
            elif (np.isfinite(edges[i]) and np.isfinite(complexity[i])
                  and np.isfinite(entry_sizes[i]) and np.isfinite(closes[i])
                  and closes[i] > 0 and abs(edges[i]) >= params.entry_strength):
                direction = 1 if edges[i] > 0 else -1
                active_size = float(entry_sizes[i])
                entry_bar = -1
            signal[i] = direction
            sizes[i] = active_size if direction else 1.0
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": 100.0, "high": 106.0, "low": 94.0,
        "close": [100.0, 100.0, 106.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    features = pd.DataFrame({
        "edge": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "complexity": 0.3, "entry_size": 0.6,
    }, index=index)
    strategy = GeneratedStrategy()
    params = OrdinalRegimeParams(hold_bars=3, profit_target=0.05)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    target = strategy.generate_signals(data, features, ctx, params).data
    assert target["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert target["size"].iloc[1] == 0.6
    data["close"] = 100.0
    timed = strategy.generate_signals(data, features, ctx, params).data
    assert timed["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    features["edge"] = -features["edge"]
    data.loc[index[2], "close"] = 94.0
    short = strategy.generate_signals(data, features, ctx, params).data
    assert short["signal"].tolist() == [0, -1, -1, 0, 0, 0, 0]
    derived = strategy.indicators(data, OrdinalRegimeParams())
    assert derived.index.equals(data.index)
    assert derived["edge"].isna().all()


if __name__ == "__main__":
    _self_check()
