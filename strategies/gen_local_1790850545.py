from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 5
    atr_period: int = 14
    entry_share: float = 0.80
    exit_share: float = 0.60
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790850545"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return max(200, 16 + params.energy_window - 1, params.atr_period + 1)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if not (params.energy_window >= 1 and params.atr_period >= 1
                and params.max_hold >= 1 and params.trail_atr > 0
                and 0 <= params.exit_share < params.entry_share <= 1):
            raise ValueError("Invalid wavelet strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        log_close = np.log(close.where(close > 0))
        out["log_close"] = log_close
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        previous = close.shift(1)
        out["previous_close"] = previous
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        for half in (1, 2, 4, 8):
            # Causal Haar detail over two adjacent, non-overlapping halves.
            average = log_close.rolling(half, min_periods=half).mean()
            out[f"mean_{half}"] = average
            out[f"detail_{half}"] = np.sqrt(half / 2.0) * (
                average - average.shift(half)
            )
            out[f"energy_{half}"] = out[f"detail_{half}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out[
            ["energy_1", "energy_2", "energy_4", "energy_8"]
        ].sum(axis=1, min_count=4)
        out["coarse_share"] = (
            (out["energy_4"] + out["energy_8"])
            / out["total_energy"].where(out["total_energy"] > 0)
        )
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveletParams
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        share = indicators["coarse_share"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        d4 = indicators["detail_4"].to_numpy(dtype=float)
        d8 = indicators["detail_8"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        peak = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = (
                np.isfinite(close[i]) and np.isfinite(atr[i])
                and np.isfinite(share[i]) and np.isfinite(ma[i])
                and np.isfinite(d4[i]) and np.isfinite(d8[i])
                and close[i] > 0 and atr[i] > 0
            )
            if active:
                held += 1
                if valid:
                    # First update is on the fill bar, never the decision bar.
                    peak = max(peak, close[i])
                    stop = max(stop, peak - params.trail_atr * atr[i])
                if (not valid or close[i] <= stop or close[i] <= ma[i]
                        or share[i] <= params.exit_share
                        or held >= params.max_hold):
                    active = False
            elif (valid and close[i] > ma[i]
                  and share[i] >= params.entry_share
                  and d4[i] > 0 and d8[i] > 0):
                active = True
                held = 0
                peak = -np.inf
                stop = -np.inf
            signal[i] = int(active)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Check that an expanding ATR cannot loosen the stop, and fills lag one bar.
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 101., 105., 104., 102., 101.]}, index=index)
    derived = pd.DataFrame({
        "atr": [1., 1., 1., 5., 5., 5.], "coarse_share": 0.9,
        "ma200": 90., "detail_4": 1., "detail_8": 1.
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, derived, None, WaveletParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    assert result.data["size"].gt(0).all()
