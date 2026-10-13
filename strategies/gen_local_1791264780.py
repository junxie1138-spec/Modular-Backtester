from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    regime_window: int = 10
    history_window: int = 126
    momentum_window: int = 5
    atr_window: int = 14
    percentile: float = 0.80
    stop_atr: float = 2.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791264780"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            max(params.regime_window, params.momentum_window) + params.history_window + 2,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.regime_window, params.history_window, params.momentum_window,
               params.atr_window, params.hold_bars) < 1:
            raise ValueError("Window lengths and hold_bars must be positive.")
        if not 0.5 < params.percentile < 1.0 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("Require 0.5 < percentile < 1 and a positive finite stop_atr.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["loss"] = (-out["return"]).clip(lower=0)
        out["loss_total"] = out["loss"].rolling(params.regime_window).sum()
        out["loss_energy"] = out["loss"].pow(2).rolling(params.regime_window).sum()
        out["concentration"] = out["loss_energy"].div(out["loss_total"].pow(2).replace(0, np.nan))
        history = out["concentration"].shift(1).rolling(params.history_window)
        out["concentration_low"] = history.quantile(1.0 - params.percentile)
        out["concentration_high"] = history.quantile(params.percentile)
        out["momentum"] = out["return"].rolling(params.momentum_window).sum()
        out["momentum_high"] = out["momentum"].shift(1).rolling(params.history_window).quantile(params.percentile)
        out["return_low"] = out["return"].shift(1).rolling(params.history_window).quantile(1.0 - params.percentile)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["elastic_entry"] = (
            (out["concentration"] >= out["concentration_high"])
            & (out["return"].shift(1) <= out["return_low"].shift(1))
            & (out["return"].shift(1) < 0)
            & (out["return"] > 0)
        )
        out["plastic_entry"] = (
            (out["concentration"] <= out["concentration_low"])
            & (out["momentum"] >= out["momentum_high"])
            & (out["momentum"] > 0)
            & (out["return"] > 0)
        )
        required = ["concentration", "concentration_low", "concentration_high",
                    "momentum", "momentum_high", "return_low", "atr"]
        out["ready"] = np.isfinite(out[required]).all(axis=1) & (out["atr"] > 0)
        out["entry"] = ((out["elastic_entry"] | out["plastic_entry"]) & out["ready"]).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        pending = False
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    entry_bar = i
                    # Entry uses the next open; ATR is frozen at the decision close.
                    # ponytail: OHLCV open approximates fill; use execution fills if the framework exposes them.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
            if entry_bar >= 0:
                if (not np.isfinite(closes[i]) or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    entry_bar = -1
                    stop = np.nan
                else:
                    raw[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                raw[i] = 1
                pending = True
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({
        "open": [100.0] + [105.0] * 11,
        "high": 111.0, "low": 99.0, "close": 105.0, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({"atr": 1.0, "entry": [True] + [False] * 11}, index=index)
    strategy = GeneratedStrategy()
    params = Params()
    data.iloc[2, data.columns.get_loc("close")] = 102.0
    stopped = strategy.generate_signals(data, ind, None, params).data["signal"]
    assert stopped.tolist() == [0, 1, 1] + [0] * 9
    data["close"] = 105.0
    data.iloc[2, data.columns.get_loc("close")] = 110.0
    held = strategy.generate_signals(data, ind, None, params).data
    assert held["signal"].tolist() == [0] + [1] * 8 + [0] * 3
    assert held["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
