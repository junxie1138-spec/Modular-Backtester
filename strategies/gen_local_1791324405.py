from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRejectionParams:
    gap_lookback: int = 126
    gap_percentile: float = 0.90
    atr_period: int = 14
    min_retrace: float = 0.30
    hold_bars: int = 8
    stop_atr: float = 2.0
    cooldown_bars: int = 3


class GeneratedStrategy(BaseStrategy[GapRejectionParams]):
    strategy_id = "gen_local_1791324405"

    @classmethod
    def params_type(cls) -> type[GapRejectionParams]:
        return GapRejectionParams

    def warmup_bars(self, params: GapRejectionParams) -> int:
        return int(params.atr_period + params.gap_lookback + 2)

    def indicators(self, data: pd.DataFrame, params: GapRejectionParams) -> pd.DataFrame:
        if params.atr_period < 1 or params.gap_lookback < 2:
            raise ValueError("Lookbacks must be positive; gap_lookback must exceed one.")
        if not 0 < params.gap_percentile < 1 or not 0 < params.min_retrace < 0.85:
            raise ValueError("Invalid percentile or retracement threshold.")
        if params.hold_bars < 1 or params.stop_atr <= 0 or params.cooldown_bars < 0:
            raise ValueError("Invalid holding, stop, or cooldown parameter.")
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()
        gap = data["open"] - previous_close
        normalized_gap = gap / atr.shift(1).replace(0.0, np.nan)
        threshold = normalized_gap.abs().rolling(
            params.gap_lookback, min_periods=params.gap_lookback
        ).quantile(params.gap_percentile).shift(1)
        retracement = (data["open"] - data["close"]) / gap.replace(0.0, np.nan)
        return pd.DataFrame({
            "atr": atr,
            "normalized_gap": normalized_gap,
            "gap_threshold": threshold,
            "retracement": retracement,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRejectionParams,
    ) -> SignalFrame:
        eligible = (
            indicators.notna().all(axis=1)
            & (indicators["atr"] > 0)
            & (indicators["normalized_gap"].abs() > indicators["gap_threshold"])
            & (indicators["retracement"] >= params.min_retrace)
            & (indicators["retracement"] <= 0.85)
        )
        candidates = pd.Series(0, index=data.index, dtype=int)
        candidates.loc[eligible & (indicators["normalized_gap"] < 0)] = 1
        candidates.loc[eligible & (indicators["normalized_gap"] > 0)] = -1
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = candidates.to_numpy(dtype=int)
        position = 0
        pending_entry = False
        entry_atr = 0.0
        fixed_stop = np.nan
        held = 0
        next_entry_bar = 0

        for i in range(len(data)):
            if pending_entry:
                # The preceding close's order fills at this bar's open.
                fixed_stop = opens[i] - position * params.stop_atr * entry_atr
                pending_entry = False
                held = 0
            if position:
                held += 1
                stopped = (
                    closes[i] < fixed_stop if position == 1
                    else closes[i] > fixed_stop
                )
                if stopped or held >= params.hold_bars:
                    position = 0
                    next_entry_bar = i + params.cooldown_bars + 1
            elif i >= next_entry_bar and entries[i] != 0:
                position = int(entries[i])
                entry_atr = float(atrs[i])
                pending_entry = True
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    opens = np.full(30, 100.0)
    closes = np.full(30, 100.0)
    opens[10], closes[10] = 90.0, 94.0
    opens[11], closes[11] = 95.0, 89.0
    opens[16], closes[16] = 110.0, 106.0
    opens[17], closes[17] = 105.0, 112.0
    data = pd.DataFrame({
        "open": opens,
        "high": np.maximum(opens, closes) + 1.0,
        "low": np.minimum(opens, closes) - 1.0,
        "close": closes,
        "volume": 1000000.0,
    }, index=index)
    params = GapRejectionParams(
        gap_lookback=5, atr_period=3, stop_atr=1.0, cooldown_bars=1
    )
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(
        data, strategy.indicators(data, params), None, params
    ).data
    assert result["signal"].iloc[10:13].tolist() == [0, 1, 0]
    assert result["signal"].iloc[16:19].tolist() == [0, -1, 0]
    assert result["signal"].isin([-1, 0, 1]).all()
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
