from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeParams:
    range_window: int = 10
    atr_multiple: float = 3.0


class GeneratedStrategy(BaseStrategy[RangeParams]):
    strategy_id = "gen_local_1791265539"

    @classmethod
    def params_type(cls):
        return RangeParams

    def warmup_bars(self, params: RangeParams) -> int:
        return int(max(params.range_window + 1, 25))

    def indicators(self, data: pd.DataFrame, params: RangeParams) -> pd.DataFrame:
        if params.range_window < 2 or not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("Require range_window >= 2 and finite atr_multiple > 0")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["previous_close"] = data["close"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["upper_extension"] = (data["high"] - out["previous_high"]).clip(lower=0)
        out["lower_extension"] = (out["previous_low"] - data["low"]).clip(lower=0)
        out["overlap"] = (
            pd.concat([data["high"], out["previous_high"]], axis=1).min(axis=1)
            - pd.concat([data["low"], out["previous_low"]], axis=1).max(axis=1)
        ).clip(lower=0).where(out["previous_high"].notna())
        out["overlap_fraction"] = out["overlap"] / out["range"].replace(0, np.nan)
        w = params.range_window
        out["upper_sum"] = out["upper_extension"].rolling(w, min_periods=w).sum()
        out["lower_sum"] = out["lower_extension"].rolling(w, min_periods=w).sum()
        out["upward_share"] = out["upper_sum"] / (
            out["upper_sum"] + out["lower_sum"]
        ).replace(0, np.nan)
        out["mean_overlap"] = out["overlap_fraction"].rolling(w, min_periods=w).mean()
        out["mean_range"] = out["range"].rolling(w, min_periods=w).mean()
        out["displacement"] = data["close"] - data["close"].shift(w)
        out["reference_range"] = out["range"].rolling(20, min_periods=20).mean().shift(1)
        out["spike"] = (out["range"] > 2.5 * out["reference_range"]).astype(float).where(
            out["reference_range"].notna()
        )
        out["recent_spikes"] = out["spike"].rolling(5, min_periods=5).sum()
        out["entry"] = (
            (out["upward_share"] >= 0.72)
            & (out["mean_overlap"] >= 0.40)
            & (out["displacement"] >= 1.5 * out["mean_range"])
            & out["recent_spikes"].eq(0)
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RangeParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        active = False
        peak = -np.inf
        stop = -np.inf
        held = 0
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                active = False
                continue
            if active:
                # Entry decision fills today; only in-trade closes update the peak.
                held += 1
                peak = max(peak, close)
                stop = max(stop, peak - params.atr_multiple * atr)
                if close <= stop or held >= 18:
                    active = False
            elif entries[i]:
                active = True
                peak = -np.inf
                stop = -np.inf
                held = 0
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RangeParams(atr_multiple=2.0)
    index = pd.date_range("2020-01-01", periods=24)
    data = pd.DataFrame({"close": [100.0, 102.0, 101.0, 99.0] + [99.0] * 20}, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 10.0, 1.0] + [1.0] * 20,
        "entry": [True] + [False] * 23,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=25)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1] + [0] * 20
    data["close"] = 100.0
    indicators["atr"] = 1.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0] + [1] * 18 + [0] * 5
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
