from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    z_window: int = 20
    atr_window: int = 14
    entry_z: float = 0.8
    regime_drift: float = 0.6
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791246281"

    @classmethod
    def params_type(cls):
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return int(max(210, params.z_window + 4, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if (params.z_window < 2 or params.atr_window < 1
                or params.entry_z <= 0 or params.regime_drift < 0
                or params.breakeven_pct <= 0 or params.trail_atr <= 0
                or params.max_hold < 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["anchor"] = close.rolling(params.z_window, min_periods=params.z_window).mean()
        out["dispersion"] = close.rolling(params.z_window, min_periods=params.z_window).std(ddof=0)
        out["z"] = (close - out["anchor"]) / out["dispersion"].replace(0, np.nan)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["drift"] = out["ma200"].diff(10) / out["atr"].replace(0, np.nan)
        out["previous_z"] = out["z"].shift(1)
        out["shelf"] = (out["z"].gt(0) & out["z"].lt(params.entry_z)).astype(int)
        out["shelf_count"] = out["shelf"].rolling(3, min_periods=3).sum().shift(1)
        out["fast_regime"] = out["drift"].ge(params.regime_drift)
        out["breakout"] = out["shelf_count"].eq(3) & out["z"].ge(params.entry_z)
        out["recovery"] = (
            out["previous_z"].le(-params.entry_z)
            & out["z"].gt(-params.entry_z) & out["z"].lt(0)
        )
        out["ready"] = (
            np.isfinite(out[["ma200", "z", "atr", "drift"]]).all(axis=1)
            & out["atr"].gt(0) & close.gt(out["ma200"])
        )
        out["entry"] = (
            out["ready"] & (
                (out["fast_regime"] & out["breakout"])
                | (~out["fast_regime"] & out["recovery"])
            )
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        n = len(data)
        decision = np.zeros(n, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma200 = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry = peak = stop = 0.0
        age = 0
        for i in range(n):
            if pending:
                pending = False
                # Entry reference is the next bar's open, never the decision close.
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    entry = peak = opens[i]
                    stop = entry - params.trail_atr * atr[i - 1]
                    armed = False
                    age = 0
            if active:
                age += 1
                valid = np.isfinite([lows[i], closes[i], atr[i], ma200[i]]).all() and atr[i] > 0
                # Test the existing stop before raising it using today's close.
                # A stop breach requests an exit at the next bar, not an intrabar fill.
                if (not valid or lows[i] <= stop
                        or closes[i] <= ma200[i] or age >= params.max_hold):
                    active = False
                    continue
                peak = max(peak, closes[i])
                armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                decision[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                decision[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decision, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100., 100., 103., 103., 100., 100.],
        "high": [101., 104., 105., 104., 101., 101.],
        "low": [99., 99., 102., 101., 99., 99.],
        "close": [100., 103., 104., 102., 100., 100.],
        "volume": [1000.] * 6,
    }, index=index)
    ind = pd.DataFrame({
        "atr": [1.] * 6, "ma200": [90.] * 6,
        "entry": [True, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, ind, None, StrategyParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert result.data["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(StrategyParams()) >= 210


if __name__ == "__main__":
    _self_check()
