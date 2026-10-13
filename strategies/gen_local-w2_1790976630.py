from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapArrivalParams:
    gap_window: int = 4
    volume_window: int = 16
    atr_window: int = 14
    min_gap_atr: float = 0.15
    volume_ratio: float = 1.10
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapArrivalParams]):
    strategy_id = "gen_local-w2_1790976630"

    @classmethod
    def params_type(cls):
        return GapArrivalParams

    def warmup_bars(self, params: GapArrivalParams) -> int:
        return int(max(params.atr_window + 2 * params.gap_window + 1,
                       params.volume_window + params.gap_window,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapArrivalParams) -> pd.DataFrame:
        if min(params.gap_window, params.volume_window, params.atr_window,
               params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not all(np.isfinite(v) and v > 0 for v in
                   (params.min_gap_atr, params.volume_ratio, params.stop_atr)):
            raise ValueError("Thresholds must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - prior_close).abs(),
             (data["low"] - prior_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["gap"] = data["open"] - prior_close
        valid_gap = out["gap"].notna() & out["prior_atr"].gt(0)
        out["up_arrival"] = (
            out["gap"] > params.min_gap_atr * out["prior_atr"]
        ).astype(float).where(valid_gap)
        out["recent_arrivals"] = out["up_arrival"].rolling(params.gap_window).sum()
        out["previous_arrivals"] = out["recent_arrivals"].shift(params.gap_window)
        out["recent_volume"] = data["volume"].rolling(params.gap_window).mean()
        out["baseline_volume"] = data["volume"].shift(params.gap_window).rolling(
            params.volume_window
        ).mean()
        out["participation"] = out["recent_volume"] / out["baseline_volume"].where(
            out["baseline_volume"] > 0
        )
        out["gap_agrees"] = (
            out["recent_arrivals"].ge(np.ceil(0.75 * params.gap_window))
            & out["previous_arrivals"].le(np.floor(0.25 * params.gap_window))
        )
        out["volume_agrees"] = out["participation"].ge(params.volume_ratio)
        out["entry"] = (
            out["gap_agrees"] & out["volume_agrees"]
            & out["atr"].gt(0) & np.isfinite(out["atr"])
            & np.isfinite(out["participation"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapArrivalParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        fixed_stop = np.nan
        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if not active and i > 0 and raw[i - 1] == 1:
                active = True
                entry_bar = i
                fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                if closes[i] < fixed_stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                else:
                    raw[i] = 1
            elif entry[i]:
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapArrivalParams()
    index = pd.date_range("2020-01-01", periods=22)
    data = pd.DataFrame(
        {"open": 100.0, "high": 111.0, "low": 93.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    data.iloc[1, data.columns.get_loc("close")] = 110.0
    data.iloc[3, data.columns.get_loc("close")] = 94.0
    indicators = pd.DataFrame({"atr": 2.0, "entry": False}, index=index)
    indicators.iloc[0, indicators.columns.get_loc("entry")] = True
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[:5].tolist() == [0, 1, 1, 1, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19:].eq(0).all()
    assert result["size"].gt(0).all()
    warm = strategy.indicators(data, params)
    assert not warm["entry"].any()


if __name__ == "__main__":
    _self_check()
