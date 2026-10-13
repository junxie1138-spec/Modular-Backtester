from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    baseline_bars: int = 64
    concentration: float = 0.8

    def __post_init__(self):
        if self.baseline_bars < 16:
            raise ValueError("baseline_bars must be at least 16")
        if not np.isfinite(self.concentration) or self.concentration <= 0:
            raise ValueError("concentration must be finite and positive")


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791285557"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(params.baseline_bars + 16)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_price"] = np.log(data["close"].where(data["close"] > 0))
        # Trailing orthonormal Haar filters; the eight-bar band has four-bar halves.
        for width in (2, 4, 8, 16):
            half = width // 2
            out[f"mean_{half}"] = out["log_price"].rolling(half).mean()
            out[f"haar_{width}"] = np.sqrt(half / 2.0) * (
                out[f"mean_{half}"] - out[f"mean_{half}"].shift(half)
            )
            out[f"energy_{width}"] = out[f"haar_{width}"].pow(2)
        out["background"] = out["energy_8"].rolling(params.baseline_bars).mean().shift(1)
        out["other_energy"] = out["energy_2"] + out["energy_4"] + out["energy_16"]
        out["concentration"] = out["energy_8"] / out["other_energy"].where(out["other_energy"] > 1e-16)
        out["shock"] = out["energy_8"] / out["background"].where(out["background"] > 1e-16)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        eligible = (
            indicators["concentration"].ge(params.concentration)
            & indicators["shock"].ge(1.5)
            & indicators["haar_8"].notna()
            & indicators["haar_8"].ne(0)
        ).fillna(False)
        entry = np.where(eligible, np.where(indicators["haar_8"] > 0, -1, 1), 0)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(entry):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                # A mirrored entry closes the trade; no same-bar reversal.
                position = 0
            raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveletParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    index = pd.date_range("2020-01-01", periods=240)
    close = 100 * np.exp(0.02 * np.sin(np.arange(240) * 2 * np.pi / 10))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1000.0}, index=index)
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(full.iloc[:180], strategy.indicators(data.iloc[:180], params))
    result = strategy.generate_signals(data, full, ctx, params).data
    prefix = strategy.generate_signals(data.iloc[:180], full.iloc[:180], ctx, params).data
    pd.testing.assert_frame_equal(result.iloc[:180], prefix)
    assert set(result["signal"]).issubset({-1, 0, 1})
    assert (result["size"] > 0).all()
    assert result["signal"].iloc[:params.baseline_bars + 8].eq(0).all()
    flat = data.copy()
    flat["close"] = 100.0
    assert strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data["signal"].eq(0).all()
    candidates = np.array([1, 0, 1, -1, -1, 0, 1, 1, 0, 0])
    toy = pd.DataFrame({"haar_8": -candidates, "concentration": 2.0,
                        "shock": 2.0}, index=index[:10])
    actual = strategy.generate_signals(data.iloc[:10], toy, ctx, params).data["signal"]
    assert actual.tolist() == [0, 1, 1, 1, 0, -1, -1, 0, 1, 1]


if __name__ == "__main__":
    _self_check()
