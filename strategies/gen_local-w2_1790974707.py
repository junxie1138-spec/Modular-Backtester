from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletSeasonParams:
    energy_floor: float = 0.60
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletSeasonParams]):
    strategy_id = "gen_local-w2_1790974707"

    @classmethod
    def params_type(cls):
        return WaveletSeasonParams

    def warmup_bars(self, params: WaveletSeasonParams) -> int:
        return 32

    def indicators(self, data: pd.DataFrame, params: WaveletSeasonParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        close = close.where(np.isfinite(close) & (close > 0))
        out = pd.DataFrame(index=data.index)
        out["log_return"] = np.log(close).diff()
        for width in (4, 8, 16):
            half = width // 2
            recent = out["log_return"].rolling(half, min_periods=half).mean()
            previous = recent.shift(half)
            out[f"haar_{width}"] = np.sqrt(half / 2.0) * (recent - previous)
            out[f"energy_{width}"] = out[f"haar_{width}"].pow(2).rolling(
                16, min_periods=16
            ).mean()
        total = out[["energy_4", "energy_8", "energy_16"]].sum(axis=1, min_count=3)
        out["energy_share"] = out["energy_16"] / total.where(total > 0)
        out["slow_drift"] = out["log_return"].rolling(8, min_periods=8).mean()
        day = data.index.day
        out["month_turn"] = (day >= 25) | (day <= 3)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletSeasonParams,
    ) -> SignalFrame:
        if not np.isfinite(params.energy_floor) or not 0 <= params.energy_floor <= 1:
            raise ValueError("energy_floor must be between zero and one")
        if isinstance(params.hold_bars, bool) or not isinstance(params.hold_bars, int) or params.hold_bars < 1:
            raise ValueError("hold_bars must be a positive integer")
        entries = (
            indicators["month_turn"].fillna(False).astype(bool)
            & (indicators["energy_share"] >= params.energy_floor)
            & (indicators["haar_16"] > 0)
            & (indicators["slow_drift"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
                # Reserve the exit bar for a flat target, even if entry qualifies.
                continue
            if entries[i]:
                exit_bar = i + params.hold_bars
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = WaveletSeasonParams()
    index = pd.date_range("2020-01-01", periods=44, freq="B")
    data = pd.DataFrame(index=index)
    features = pd.DataFrame(
        {"month_turn": True, "energy_share": 0.9, "haar_16": 1.0, "slow_drift": 0.01},
        index=index,
    )
    result = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    expected = np.zeros(44, dtype=int)
    expected[1:19] = 1
    expected[20:38] = 1
    expected[39:44] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert (result["size"] > 0).all()
    features["energy_share"] = np.nan
    flat = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert (flat["signal"] == 0).all()


if __name__ == "__main__":
    _self_check()
