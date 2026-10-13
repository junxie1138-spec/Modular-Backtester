from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletReleaseParams:
    energy_window: int = 3
    release_ratio: float = 1.4
    breakout_window: int = 2


class GeneratedStrategy(BaseStrategy[WaveletReleaseParams]):
    strategy_id = "gen_local-w2_1791259765"

    @classmethod
    def params_type(cls) -> type[WaveletReleaseParams]:
        return WaveletReleaseParams

    def warmup_bars(self, params: WaveletReleaseParams) -> int:
        return int(max(8 + params.energy_window, params.breakout_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletReleaseParams) -> pd.DataFrame:
        if params.energy_window < 1 or params.breakout_window < 1 or not np.isfinite(params.release_ratio) or params.release_ratio <= 1.0:
            raise ValueError("Positive windows and release_ratio > 1 are required")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for scale in (1, 2, 4):
            mean = out["log_close"].rolling(scale, min_periods=scale).mean()
            out[f"haar_{scale}"] = np.sqrt(scale / 2.0) * (mean - mean.shift(scale))
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
        out["total_energy"] = out[["energy_1", "energy_2", "energy_4"]].sum(axis=1, min_count=3)
        out["fast_share"] = out["energy_1"] / out["total_energy"].where(out["total_energy"] > 0)
        out["prior_fast_share"] = out["fast_share"].shift(1)
        out["prior_fast_energy"] = out["energy_1"].shift(1)
        out["resistance"] = data["high"].rolling(params.breakout_window, min_periods=params.breakout_window).max().shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletReleaseParams) -> SignalFrame:
        ready = indicators[["fast_share", "prior_fast_share", "energy_1", "prior_fast_energy", "resistance"]].notna().all(axis=1)
        entry = (
            ready
            & (indicators["fast_share"] > params.release_ratio * indicators["prior_fast_share"])
            & (indicators["energy_1"] > indicators["prior_fast_energy"])
            & (data["close"] > indicators["resistance"])
        )
        df = pd.DataFrame(index=data.index)
        # Symmetric reversal: the identical condition enters when true and exits when false.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = WaveletReleaseParams()
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.Series(100.0 + np.arange(40) * 0.1 + np.sin(np.arange(40)), index=index)
    data = pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1000.0})
    actual = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:25], params)
    pd.testing.assert_frame_equal(actual.iloc[:25], prefix)
    assert not actual.iloc[:strategy.warmup_bars(params) - 1]["prior_fast_share"].notna().any()
    probe = pd.DataFrame({"fast_share": [0.8, 0.2, 0.8], "prior_fast_share": [0.2, 0.8, 0.2], "energy_1": [2.0, 1.0, 2.0], "prior_fast_energy": [1.0, 2.0, 1.0], "resistance": [99.0, 99.0, 99.0]}, index=index[:3])
    signals = strategy.generate_signals(data.iloc[:3], probe, None, params).data
    assert signals["signal"].tolist() == [0, 1, 0]
    assert (signals["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
