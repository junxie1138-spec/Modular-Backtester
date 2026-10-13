from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_days: int = 3
    load_days: int = 5
    atr_days: int = 14
    shock_atr: float = 0.75
    overload_ratio: float = 1.25
    stop_atr: float = 1.8
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791234590"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.atr_days + 1,
                       params.roc_days + params.load_days + 2))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.roc_days, params.load_days, params.atr_days) < 1:
            raise ValueError("Indicator lookbacks must be positive")
        if params.hold_days < 1 or min(params.shock_atr, params.overload_ratio, params.stop_atr) <= 0:
            raise ValueError("Holding period and thresholds must be positive")
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["roc"] = close.pct_change(params.roc_days, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_days, min_periods=params.atr_days).mean()
        out["atr_fraction"] = (out["atr"] / close).where(out["atr"] > 0)
        out["prior_sell_load"] = (-out["acceleration"]).clip(lower=0).shift(1).rolling(
            params.load_days, min_periods=params.load_days
        ).sum()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (close > out["ma200"])
            & (out["roc"] < 0)
            & (-out["acceleration"] > params.shock_atr * out["atr_fraction"])
            & (-out["acceleration"] > params.overload_ratio * out["prior_sell_load"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_atr = 0.0
        stop = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # The order decided yesterday fills at today's open.
                    # ATR is frozen on the decision bar; this stop never ratchets.
                    stop = opens[i] - params.stop_atr * entry_atr
                if (not np.isfinite(close[i]) or not np.isfinite(stop)
                        or close[i] < stop
                        or i - entry_bar + 1 >= params.hold_days):
                    entry_bar = -1
                    continue
                signal[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                entry_bar = i + 1
                entry_atr = atr[i]
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 96.0,
        "close": [100.0, 103.0, 101.0, 99.0, 97.0, 100.0, 100.0, 100.0],
        "volume": 1000.0
    }, index=index)
    ind = pd.DataFrame({"atr": 1.0, "entry": [True] + [False] * 7}, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, ind, ctx, Params(stop_atr=2.0, hold_days=5)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, ind, ctx, Params(hold_days=4)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
