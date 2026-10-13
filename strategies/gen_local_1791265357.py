from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapWaveletParams:
    gap_atr: float = 0.65
    fine_share: float = 0.50
    decay_fraction: float = 0.25
    trail_atr: float = 2.5
    atr_period: int = 14
    baseline_period: int = 20
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[GapWaveletParams]):
    strategy_id = "gen_local_1791265357"

    @classmethod
    def params_type(cls) -> type[GapWaveletParams]:
        return GapWaveletParams

    def warmup_bars(self, params: GapWaveletParams) -> int:
        return int(max(200, params.atr_period + 1, params.baseline_period + 9))

    def indicators(self, data: pd.DataFrame, params: GapWaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = data["open"].div(out["previous_close"].replace(0, np.nan)) - 1.0
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        # Causal Haar details: adjacent blocks of 1, 2, and 4 overnight gaps.
        for half in (1, 2, 4):
            mean = out["gap"].rolling(half, min_periods=half).mean()
            out[f"gap_mean_{half}"] = mean
            out[f"detail_{half}"] = np.sqrt(half / 2.0) * (mean - mean.shift(half))
            out[f"energy_{half}"] = out[f"detail_{half}"].pow(2)
        out["total_energy"] = out[["energy_1", "energy_2", "energy_4"]].sum(axis=1, min_count=3)
        out["energy_baseline"] = out["total_energy"].rolling(
            params.baseline_period, min_periods=params.baseline_period
        ).mean().shift(1)
        out["fine_fraction"] = out["energy_1"].div(out["total_energy"].replace(0, np.nan))
        out["gap_depth"] = (out["previous_close"] - data["open"]).div(
            out["previous_atr"].replace(0, np.nan)
        )
        out["repair_fraction"] = (data["close"] - data["open"]).div(
            (out["previous_close"] - data["open"]).replace(0, np.nan)
        )
        out["shock"] = (
            (out["gap_depth"] >= params.gap_atr)
            & (out["repair_fraction"] >= 0.5)
            & (out["fine_fraction"] >= params.fine_share)
            & (out["total_energy"] > 1.5 * out["energy_baseline"])
        ).fillna(False)
        out["ready"] = (
            np.isfinite(out[["atr", "ma200", "energy_1", "energy_baseline"]]).all(axis=1)
            & (out["atr"] > 0)
            & np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
        )
        out["entry"] = (
            out["ready"]
            & out["shock"].shift(2, fill_value=False)
            & (out["energy_1"] <= params.decay_fraction * out["energy_1"].shift(2))
            & (data["close"] > data["close"].shift(2))
            & (data["close"] > data["open"])
            & (data["close"] > out["ma200"])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapWaveletParams
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = -np.inf
        stop = -np.inf
        age = 0
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        regimes = indicators["ma200"].to_numpy(dtype=float)
        ready = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if active:
                # Entry decision was yesterday; this is an actual held close.
                age += 1
                if not ready[i]:
                    active = False
                else:
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atrs[i])
                    if closes[i] < stop or closes[i] <= regimes[i] or age >= params.max_hold:
                        active = False
            elif ready[i] and entries[i]:
                active = True
                high_water = -np.inf
                stop = -np.inf
                age = 0
            signals[i] = int(active)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    data = pd.DataFrame({"close": [150.0, 110.0, 108.0, 105.0, 104.0, 103.0]}, index=index)
    features = pd.DataFrame({
        "atr": [2.0, 2.0, 10.0, 10.0, 10.0, 10.0],
        "ma200": 90.0, "ready": True,
        "entry": [True, False, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    params = GapWaveletParams(trail_atr=2.0)
    result = strategy.generate_signals(data, features, None, params).data
    # Uses held closes only, shifts once, and retains the 106 stop as ATR widens.
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
