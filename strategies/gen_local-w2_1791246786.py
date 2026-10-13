from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapReleaseParams:
    energy_window: int = 5
    atr_window: int = 14
    percentile_window: int = 126
    tension_percentile: float = 0.85
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapReleaseParams]):
    strategy_id = "gen_local-w2_1791246786"

    @classmethod
    def params_type(cls) -> type[GapReleaseParams]:
        return GapReleaseParams

    @staticmethod
    def warmup_bars(params: GapReleaseParams) -> int:
        return int(max(params.energy_window, params.atr_window) + params.percentile_window + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapReleaseParams) -> pd.DataFrame:
        if min(params.energy_window, params.atr_window, params.percentile_window) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not 0.0 < params.tension_percentile < 1.0:
            raise ValueError("tension_percentile must be between zero and one")
        if params.profit_target <= 0.0 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Use a positive profit target and a three-to-five-bar holding limit")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = data["open"] - out["previous_close"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["downside_gap_squared"] = out["gap"].clip(upper=0.0).pow(2)
        out["downside_gap_rms"] = np.sqrt(
            out["downside_gap_squared"].rolling(params.energy_window).mean()
        )
        out["tension"] = (
            out["downside_gap_rms"] / out["atr"].where(out["atr"] > 0.0)
        ).replace([np.inf, -np.inf], np.nan)
        out["tension_threshold"] = out["tension"].rolling(
            params.percentile_window
        ).quantile(params.tension_percentile).shift(1)
        out["negative_gap"] = out["gap"].lt(0.0).astype(float).where(out["gap"].notna())
        out["negative_gap_count"] = out["negative_gap"].rolling(params.energy_window).sum()
        out["prior_tension"] = out["tension"].shift(1)
        out["prior_threshold"] = out["tension_threshold"].shift(1)
        out["prior_negative_count"] = out["negative_gap_count"].shift(1)
        out["entry"] = (
            out["prior_tension"].gt(out["prior_threshold"])
            & out["prior_negative_count"].ge((params.energy_window + 1) // 2)
            & out["gap"].gt(0.0)
            & data["close"].gt(out["previous_close"])
            & data["close"].gt(data["open"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapReleaseParams,
    ) -> SignalFrame:
        if params.profit_target <= 0.0 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Use a positive profit target and a three-to-five-bar holding limit")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        held_bars = 0
        for i in range(len(data)):
            if active:
                # An entry decided yesterday executes at today's open.
                if held_bars == 0:
                    entry_price = opens[i]
                held_bars += 1
                target_hit = (
                    np.isfinite(entry_price)
                    and entry_price > 0.0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                # Both exit decisions execute at the following open.
                if target_hit or held_bars >= params.hold_bars:
                    active = False
                    entry_price = np.nan
                    held_bars = 0
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                active = True
                entry_price = np.nan
                held_bars = 0
            decisions[i] = int(active)
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
    data.loc[index[2], "close"] = 102.0
    indicators = pd.DataFrame({"entry": False}, index=index)
    indicators.loc[index[[0, 6]], "entry"] = True
    result = GeneratedStrategy.generate_signals(data, indicators, None, GapReleaseParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    assert result.data["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
