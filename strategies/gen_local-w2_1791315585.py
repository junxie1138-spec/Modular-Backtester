from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    history: int = 252
    neighbors: int = 9
    min_conviction: float = 0.20


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791315585"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        if not 2 <= params.neighbors <= params.history:
            raise ValueError("Require 2 <= neighbors <= history")
        if not 0.0 < params.min_conviction < 1.0:
            raise ValueError("min_conviction must be between zero and one")
        return int(params.history + 19)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        for half in (1, 2, 4):
            scale = 2 * half
            out[f"mean_{half}"] = out["return"].rolling(half, min_periods=half).mean()
            out[f"haar_{scale}"] = np.sqrt(half / 2.0) * (
                out[f"mean_{half}"] - out[f"mean_{half}"].shift(half)
            )
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2)
        out["total_energy"] = out[["energy_2", "energy_4", "energy_8"]].sum(axis=1, min_count=3)
        denominator = out["total_energy"].where(out["total_energy"] > 0)
        features = []
        for scale in (2, 4, 8):
            signed = f"signed_{scale}"
            delta = f"delta_{scale}"
            out[signed] = np.sign(out[f"haar_{scale}"]) * out[f"energy_{scale}"] / denominator
            out[delta] = out[signed].diff()
            features.extend((signed, delta))
        # Payoff ending today: buy the preceding session's open, sell today's close.
        out["realized_two_session"] = data["close"] / data["open"].shift(1).where(data["open"].shift(1) > 0) - 1.0
        out["ready"] = np.isfinite(out[features]).all(axis=1).astype(float)
        out["payoff_ready"] = np.isfinite(out["realized_two_session"]).astype(float)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        columns = [name for scale in (2, 4, 8) for name in (f"signed_{scale}", f"delta_{scale}")]
        shapes = indicators[columns].to_numpy(dtype=float)
        ready = indicators["ready"].to_numpy(dtype=bool)
        payoff_ready = indicators["payoff_ready"].to_numpy(dtype=bool)
        payoffs = indicators["realized_two_session"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.15, dtype=float)
        state = 0
        last_size = 0.15
        # ponytail: O(bars * history) bounded search; use batched distances if histories grow materially.
        for t in range(warmup - 1, len(data)):
            direction = 0
            conviction = None
            if ready[t]:
                # Last candidate ends at t-10: nine-bar shape plus two-session outcome do not overlap today's shape.
                candidates = np.arange(t - params.history - 9, t - 9)
                candidates = candidates[ready[candidates] & payoff_ready[candidates + 2]]
                if len(candidates) >= params.neighbors:
                    distances = np.square(shapes[candidates] - shapes[t]).sum(axis=1)
                    nearest = candidates[np.argsort(distances, kind="stable")[:params.neighbors]]
                    outcomes = payoffs[nearest + 2]
                    forecast = float(outcomes.mean())
                    rms = float(np.sqrt(np.mean(np.square(outcomes))))
                    conviction = min(1.0, abs(forecast) / rms) if rms > 0 else 0.0
                    if conviction >= params.min_conviction:
                        direction = 1 if forecast > 0 else -1
            if state == 0:
                if direction:
                    state = direction
            elif direction == -state:
                state = 0
            if conviction is not None:
                last_size = 0.15 + 0.80 * conviction
            signals[t] = state
            sizes[t] = last_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.15).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = AnalogParams(history=60, neighbors=5, min_conviction=0.15)
    index = pd.date_range("2020-01-01", periods=400, freq="B")
    t = np.arange(len(index), dtype=float)
    close = 100.0 * np.exp(np.cumsum(0.008 * np.sin(t * 0.47) + 0.003 * np.cos(t * 0.19)))
    data = pd.DataFrame({"open": close * np.exp(-0.002 * np.cos(t)), "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    full = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    prefix = data.iloc[:280]
    partial = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(full.iloc[:280], partial)
    assert full["signal"].isin([-1, 0, 1]).all() and full["signal"].ne(0).any()
    assert np.isfinite(full["size"]).all() and full["size"].between(0.15, 0.95).all()
