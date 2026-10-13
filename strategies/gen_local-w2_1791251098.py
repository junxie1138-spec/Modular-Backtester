from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletEchoParams:
    energy_window: int = 40
    entry_threshold: float = 0.35
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    atr_window: int = 14
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[WaveletEchoParams]):
    strategy_id = "gen_local-w2_1791251098"

    @classmethod
    def params_type(cls):
        return WaveletEchoParams

    def warmup_bars(self, params):
        return int(max(params.energy_window + 32, params.atr_window + 1))

    def indicators(self, data, params):
        if (params.energy_window < 3 or params.atr_window < 1
                or params.max_hold < 1 or params.entry_threshold <= 0
                or params.breakeven_pct <= 0 or params.trail_atr <= 0):
            raise ValueError("Windows, thresholds and holding period must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff()
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        numerator = pd.Series(0.0, index=data.index)
        denominator = pd.Series(0.0, index=data.index)
        for half in (2, 4, 8):
            block = out["return"].rolling(half).sum()
            detail = (block - block.shift(half)) / np.sqrt(2.0 * half)
            energy = detail.pow(2).rolling(params.energy_window).mean()
            # Nonoverlapping packets distinguish echoes from overlap-induced correlation.
            rho = detail.rolling(params.energy_window).corr(detail.shift(2 * half))
            standardized = detail / np.sqrt(energy.where(energy > 0))
            out[f"detail_{half}"] = detail
            out[f"energy_{half}"] = energy
            out[f"rho_{half}"] = rho
            out[f"echo_{half}"] = rho.clip(-1.0, 1.0) * standardized.clip(-3.0, 3.0)
            numerator = numerator + energy * out[f"echo_{half}"]
            denominator = denominator + energy
        out["score"] = (numerator / denominator.where(denominator > 0)).clip(-3.0, 3.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.2, dtype=float)
        score = indicators["score"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side = 0
        pending = False
        entry = 0.0
        stop = -np.inf
        armed = False
        age = 0
        position_size = 0.2
        for i in range(n):
            ready = (np.isfinite(score[i]) and np.isfinite(atr[i])
                     and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0)
            if side:
                if pending:
                    # The prior close's order enters at today's observed open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        continue
                    entry = opens[i]
                    stop = side * entry - params.trail_atr * atr[i - 1]
                    pending = False
                age += 1
                if not ready:
                    side = 0
                    continue
                favorable_close = side * closes[i]
                if favorable_close <= stop or age >= params.max_hold:
                    side = 0
                    continue
                if side * (closes[i] / entry - 1.0) >= params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed price makes the ratchet increase for both directions.
                    stop = max(stop, side * entry,
                               favorable_close - params.trail_atr * atr[i])
                signals[i] = side
                sizes[i] = position_size
            elif ready and abs(score[i]) >= params.entry_threshold:
                side = 1 if score[i] > 0 else -1
                pending = True
                armed = False
                age = 0
                position_size = float(0.2 + 0.75 * np.clip(abs(score[i]) / 2.0, 0.0, 1.0))
                signals[i] = side
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = WaveletEchoParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=9)
    close = np.array([100, 102, 104, 102, 99, 98, 96, 98, 98], dtype=float)
    sample = pd.DataFrame({
        "open": [100, 100, 102, 104, 102, 99, 98, 96, 98],
        "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        "score": [1, 1, 1, 1, -1, -1, -1, -1, 0],
        "atr": 1.0,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(sample, derived, ctx, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
    weaker = derived.copy()
    weaker["score"] *= 0.5
    smaller = strategy.generate_signals(sample, weaker, ctx, p).data
    assert smaller["size"].iloc[1] < result["size"].iloc[1]
