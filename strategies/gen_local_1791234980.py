from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 12
    energy_cutoff: float = 0.56
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791234980"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        if params.energy_window < 2 or params.hold_bars not in (1, 2):
            raise ValueError("energy_window must be >= 2; hold_bars must be 1 or 2")
        if not 0.5 < params.energy_cutoff < 1.0:
            raise ValueError("energy_cutoff must lie strictly between 0.5 and 1")
        return int(params.energy_window + 9)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        r = out["return"]
        for span in (2, 4, 8):
            out[f"drift_{span}"] = r.rolling(span, min_periods=span).sum()
        out["haar_2"] = (r - r.shift(1)) / np.sqrt(2.0)
        out["haar_4"] = (out["drift_2"] - out["drift_2"].shift(2)) / 2.0
        out["haar_8"] = (out["drift_4"] - out["drift_4"].shift(4)) / np.sqrt(8.0)
        for span in (2, 4, 8):
            out[f"energy_{span}"] = out[f"haar_{span}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out[["energy_2", "energy_4", "energy_8"]].sum(axis=1, min_count=3)
        denominator = out["total_energy"].where(out["total_energy"] > 0)
        out["coarse_share"] = out["energy_8"] / denominator
        out["fine_share"] = out["energy_2"] / denominator
        out["trend_regime"] = out["coarse_share"].gt(params.energy_cutoff)
        out["rebound_regime"] = out["fine_share"].gt(params.energy_cutoff)
        out["trend_setup"] = out["trend_regime"] & out["drift_8"].gt(0) & out["drift_2"].gt(0)
        out["rebound_setup"] = out["rebound_regime"] & out["drift_8"].lt(0) & out["drift_2"].gt(0)
        out["confirmed"] = (
            (out["trend_setup"] & out["trend_setup"].shift(1, fill_value=False))
            | (out["rebound_setup"] & out["rebound_setup"].shift(1, fill_value=False))
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        entries = indicators["confirmed"].fillna(False).to_numpy(dtype=bool).copy()
        entries[:warmup] = False
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
            elif enter:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = WaveletParams()
    warmup = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=warmup + 12, freq="D")
    data = pd.DataFrame({"close": 100.0}, index=index)
    confirmed = pd.DataFrame({"confirmed": True}, index=index)
    result = strategy.generate_signals(data, confirmed, None, params).data
    assert result["signal"].iloc[:warmup + 1].eq(0).all()
    assert result["signal"].iloc[warmup + 1:warmup + 7].tolist() == [1, 1, 0, 1, 1, 0]
    assert result["size"].gt(0).all()
    assert not strategy.indicators(data, params)["confirmed"].any()
