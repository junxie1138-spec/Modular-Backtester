from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangePinchParams:
    regime_window: int = 200
    equilibrium_window: int = 20
    pinch_bars: int = 3
    atr_window: int = 14
    width_ratio: float = 0.75
    displacement_atr: float = 0.5
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[RangePinchParams]):
    strategy_id = "gen_local_1790877241"

    @classmethod
    def params_type(cls) -> type[RangePinchParams]:
        return RangePinchParams

    def warmup_bars(self, params: RangePinchParams) -> int:
        return int(max(params.regime_window, params.equilibrium_window + params.pinch_bars, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RangePinchParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["width"] = data["high"] - data["low"]
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["width"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["regime_ma"] = data["close"].rolling(params.regime_window, min_periods=params.regime_window).mean()
        out["equilibrium"] = out["midpoint"].rolling(params.equilibrium_window, min_periods=params.equilibrium_window).mean().shift(params.pinch_bars)
        out["high_change"] = data["high"].diff(params.pinch_bars)
        out["low_change"] = data["low"].diff(params.pinch_bars)
        out["width_ratio"] = out["width"] / out["width"].shift(params.pinch_bars).replace(0.0, np.nan)
        out["displacement"] = (out["equilibrium"] - out["midpoint"]) / out["atr"].replace(0.0, np.nan)
        out["regime_ok"] = (data["close"] > out["regime_ma"]).fillna(False)
        out["entry_setup"] = (
            out["regime_ok"]
            & (out["high_change"] < 0.0)
            & (out["low_change"] > 0.0)
            & (out["width_ratio"] <= params.width_ratio)
            & (out["displacement"] >= params.displacement_atr)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangePinchParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=np.int64)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        setups = indicators["entry_setup"].fillna(False).to_numpy(dtype=bool)
        regimes = indicators["regime_ok"].fillna(False).to_numpy(dtype=bool)
        pending_entry = False
        active = False
        armed = False
        entry = 0.0
        stop = 0.0
        entry_atr = 0.0
        held = 0

        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    entry = opens[i]
                    stop = entry - params.trail_atr * entry_atr
                    held = 0
                    armed = False
                    active = True

            if active:
                held += 1
                close = closes[i]
                if not np.isfinite(close) or close <= 0.0:
                    decisions[i] = 1
                    continue
                if close <= stop or held >= params.max_hold_bars or not regimes[i]:
                    active = False
                    continue
                if close >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                    stop = max(stop, entry)
                if armed and np.isfinite(atrs[i]) and atrs[i] > 0.0:
                    stop = max(stop, entry, close - params.trail_atr * atrs[i])
                decisions[i] = 1
            elif setups[i] and np.isfinite(atrs[i]) and atrs[i] > 0.0:
                entry_atr = atrs[i]
                pending_entry = True
                decisions[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": [100.0] * 6, "close": [100.0, 100.0, 104.0, 103.0, 101.0, 101.0]}, index=index)
    indicators = pd.DataFrame({"atr": [1.0, 1.0, 1.0, 10.0, 10.0, 10.0], "entry_setup": [True, False, False, False, False, False], "regime_ok": [True] * 6}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, RangePinchParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    assert (result.data["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
