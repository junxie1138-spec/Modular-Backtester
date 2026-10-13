from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    atr_window: int = 14
    gap_min_atr: float = 0.15
    gap_decay: float = 0.8
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791238314"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.atr_window + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.atr_window < 1 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Require atr_window >= 1 and hold_bars between 3 and 5.")
        if not (params.gap_min_atr > 0 and 0 < params.gap_decay <= 1 and params.stop_atr > 0):
            raise ValueError("Require positive gap/stop thresholds and gap_decay in (0, 1].")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["gap"] = data["open"] - out["prior_close"]
        out["body"] = data["close"] - data["open"]
        out["gap_atr"] = out["gap"] / out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["qualified"] = (
            (out["gap_atr"] >= params.gap_min_atr)
            & (data["low"] >= out["prior_close"])
            & (out["body"] > 0)
        ).fillna(False)
        # Two accepted upside gaps: overnight impulse shrinks as daytime buying strengthens.
        out["confirmed"] = (
            out["qualified"]
            & out["qualified"].shift(1, fill_value=False)
            & (out["gap"] <= params.gap_decay * out["gap"].shift(1))
            & (out["body"] > out["body"].shift(1))
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = 0
        fixed_stop = np.nan
        for i in range(len(data)):
            # A prior close's entry decision fills at this bar's open.
            if not active and i > 0 and raw[i - 1] == 1:
                if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                active = True
                entry_bar = i
                fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                # Stop stays anchored to the entry open and pre-entry ATR.
                if not np.isfinite(closes[i]) or closes[i] < fixed_stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                    continue
                raw[i] = 1
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 96.0,
        "close": [100.0, 100.0, 100.0, 103.0, 97.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({"atr": 1.0, "confirmed": [False, True, False, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    stopped = strategy.generate_signals(data, ind, ctx, params).data
    assert stopped["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    data.loc[index[4], "close"] = 99.0
    timed = strategy.generate_signals(data, ind, ctx, params).data
    assert timed["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert (timed["size"] > 0).all()
    assert strategy.indicators(data, params)["confirmed"].eq(False).all()


if __name__ == "__main__":
    _self_check()
