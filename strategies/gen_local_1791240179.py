from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletGapParams:
    energy_window: int = 8
    recovery_threshold: float = 0.2


class GeneratedStrategy(BaseStrategy[WaveletGapParams]):
    strategy_id = "gen_local_1791240179"

    @classmethod
    def params_type(cls):
        return WaveletGapParams

    def warmup_bars(self, params: WaveletGapParams) -> int:
        return int(params.energy_window) + 8

    def indicators(self, data: pd.DataFrame, params: WaveletGapParams) -> pd.DataFrame:
        if params.energy_window < 1 or not 0.0 <= params.recovery_threshold < 1.0:
            raise ValueError("Require energy_window >= 1 and 0 <= recovery_threshold < 1.")
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["gap"] = np.log(opening / closing.shift(1)).replace([np.inf, -np.inf], np.nan)
        out["session"] = np.log(closing / opening).replace([np.inf, -np.inf], np.nan)
        directional = pd.Series(0.0, index=data.index)
        total = pd.Series(0.0, index=data.index)
        for half in (1, 2, 4):
            for channel in ("gap", "session"):
                block = out[channel].rolling(half, min_periods=half).sum()
                out[f"{channel}_haar_{half}"] = (block - block.shift(half)) / np.sqrt(2.0 * half)
            gap = out[f"gap_haar_{half}"]
            session = out[f"session_haar_{half}"]
            cross = gap * session
            # Opposing channels contribute bullish or bearish absorption energy.
            out[f"repair_energy_{half}"] = (
                -2.0 * cross.where(cross < 0, 0.0) * np.sign(session)
            ).where(gap.notna() & session.notna())
            out[f"total_energy_{half}"] = gap.pow(2) + session.pow(2)
            directional = directional + out[f"repair_energy_{half}"]
            total = total + out[f"total_energy_{half}"]
        out["repair_energy"] = directional.rolling(params.energy_window, min_periods=params.energy_window).sum()
        out["total_energy"] = total.rolling(params.energy_window, min_periods=params.energy_window).sum()
        out["repair_score"] = (out["repair_energy"] / out["total_energy"].where(out["total_energy"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["net_repair"] = (out["gap"] + out["session"]).rolling(params.energy_window, min_periods=params.energy_window).sum()
        out["ready"] = out[["repair_score", "net_repair"]].notna().all(axis=1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletGapParams,
    ) -> SignalFrame:
        # Entry and exit use exactly the same predicate: hold iff it is true.
        eligible = (
            indicators["ready"]
            & indicators["repair_score"].gt(params.recovery_threshold)
            & indicators["net_repair"].gt(0.0)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = eligible.astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=128, freq="B")
    gaps = rng.normal(0.0, 0.006, len(index))
    sessions = -0.8 * gaps + rng.normal(0.001, 0.002, len(index))
    close = 100.0 * np.exp(np.cumsum(gaps + sessions))
    opening = np.r_[100.0, close[:-1]] * np.exp(gaps)
    data = pd.DataFrame({"open": opening, "high": np.maximum(opening, close), "low": np.minimum(opening, close), "close": close, "volume": 1000000.0}, index=index)
    strategy = GeneratedStrategy()
    params = WaveletGapParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    eligible = indicators["ready"] & indicators["repair_score"].gt(params.recovery_threshold) & indicators["net_repair"].gt(0.0)
    expected = eligible.astype(int).shift(1).fillna(0).astype(int)
    assert signals["signal"].equals(expected)
    assert set(signals["signal"].unique()) == {0, 1}
    assert signals["signal"].iloc[:strategy.warmup_bars(params)].eq(0).all()
    assert signals["size"].gt(0).all()
    pd.testing.assert_frame_equal(indicators.iloc[:80], strategy.indicators(data.iloc[:80], params))
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    assert strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data["signal"].eq(0).all()
