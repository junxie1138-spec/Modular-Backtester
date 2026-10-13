from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PhaseParams:
    baseline_bars: int = 20
    phase_bars: int = 3
    volume_multiple: float = 1.3
    impulse_multiple: float = 0.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[PhaseParams]):
    strategy_id = "gen_local_1790884597"

    @classmethod
    def params_type(cls):
        return PhaseParams

    def warmup_bars(self, params: PhaseParams) -> int:
        return int(max(params.baseline_bars + 2, params.phase_bars + 1))

    def indicators(self, data: pd.DataFrame, params: PhaseParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).replace(0, np.nan)
        out["return"] = data["close"] / previous_close - 1.0
        out["overnight"] = data["open"] / previous_close - 1.0
        out["daytime"] = data["close"] / data["open"].replace(0, np.nan) - 1.0
        out["volume_baseline"] = data["volume"].rolling(
            params.baseline_bars, min_periods=params.baseline_bars
        ).mean().shift(1)
        out["move_baseline"] = out["return"].abs().rolling(
            params.baseline_bars, min_periods=params.baseline_bars
        ).mean().shift(1)
        out["relative_volume"] = data["volume"] / out["volume_baseline"].replace(0, np.nan)
        out["impulse"] = out["return"] / out["move_baseline"].replace(0, np.nan)
        out["overnight_drift"] = out["overnight"].rolling(
            params.phase_bars, min_periods=params.phase_bars
        ).sum()
        out["daytime_drift"] = out["daytime"].rolling(
            params.phase_bars, min_periods=params.phase_bars
        ).sum()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PhaseParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        ready = indicators[[
            "relative_volume", "impulse", "overnight_drift", "daytime_drift"
        ]].notna().all(axis=1)
        volume_confirmed = ready & (
            indicators["relative_volume"] >= params.volume_multiple
        )
        long_entry = volume_confirmed & (
            indicators["impulse"] >= params.impulse_multiple
        ) & (indicators["overnight_drift"] > 0) & (indicators["daytime_drift"] > 0)
        short_entry = volume_confirmed & (
            indicators["impulse"] <= -params.impulse_multiple
        ) & (indicators["overnight_drift"] < 0) & (indicators["daytime_drift"] < 0)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i == exit_bar:
                    position = 0
            elif entry != 0:
                position = int(entry)
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = PhaseParams()
    index = pd.date_range("2020-01-01", periods=80, freq="B")
    close = pd.Series(100.0 + np.arange(80) * 0.2, index=index)
    data = pd.DataFrame({
        "open": close - 0.1, "high": close + 0.5,
        "low": close - 0.5, "close": close, "volume": 1000.0,
    }, index=index)
    calculated = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(
        calculated.iloc[:40], strategy.indicators(data.iloc[:40], params)
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    assert (strategy.generate_signals(data, calculated, ctx, params).data["signal"] == 0).all()
    probe = data.iloc[:11]
    direction = np.array([1] + [-1] * 10)
    forced = pd.DataFrame({
        "relative_volume": 2.0, "impulse": direction * 2.0,
        "overnight_drift": direction * 0.01,
        "daytime_drift": direction * 0.01,
    }, index=probe.index)
    for holding in (3, 4, 5):
        params.hold_bars = holding
        for side in (1, -1):
            test = forced.copy()
            test[["impulse", "overnight_drift", "daytime_drift"]] *= side
            result = strategy.generate_signals(probe, test, ctx, params).data
            assert result["signal"].iloc[0] == 0
            assert result["signal"].iloc[1:holding + 1].tolist() == [side] * holding
            assert result["signal"].iloc[holding + 1] == 0
            assert result["signal"].iloc[holding + 2] == -side
            assert result["size"].gt(0).all()
