from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 12
    threshold_window: int = 126
    entry_percentile: float = 0.80
    atr_window: int = 14
    stop_atr: float = 2.2
    hold_bars: int = 8

    def __post_init__(self):
        if self.graph_window < 2 or self.threshold_window < 2:
            raise ValueError("Graph and percentile windows must be at least two.")
        if self.atr_window < 1 or self.hold_bars < 1:
            raise ValueError("ATR and holding windows must be positive.")
        if not 0.0 < self.entry_percentile < 1.0 or not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("Invalid percentile or stop multiplier.")


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791306943"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        return int(max(params.graph_window + params.threshold_window + 2, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_return"] = np.log(close.where(close > 0)).diff()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        returns = out["log_return"].to_numpy()
        lags = np.arange(1, params.graph_window + 1, dtype=float)
        past = np.column_stack(
            [out["log_return"].shift(int(lag)).to_numpy() for lag in lags]
        )
        # Natural visibility: every intermediate point lies below the connecting line.
        slopes = (past - returns[:, None]) / lags
        ceiling = np.column_stack(
            [np.full(len(data), -np.inf), np.maximum.accumulate(slopes, axis=1)[:, :-1]]
        )
        visible = np.isfinite(slopes) & (slopes > ceiling)
        ready = np.isfinite(returns) & np.isfinite(past).all(axis=1)
        for j, lag in enumerate(lags):
            out[f"visible_lag_{int(lag)}"] = np.where(ready, visible[:, j].astype(float), np.nan)
        # Adjacent nodes are always linked; only nonlocal links carry topology information.
        visible[:, 0] = False
        weights = visible * lags
        mass = weights.sum(axis=1)
        mean = np.divide(
            (np.where(visible, past, 0.0) * lags).sum(axis=1),
            mass, out=np.zeros(len(data)), where=mass > 0
        )
        out["nonlocal_degree"] = np.where(ready, visible.sum(axis=1), np.nan)
        out["visible_return_mean"] = np.where(ready, mean, np.nan)
        out["return_variance"] = out["log_return"].rolling(params.graph_window + 1).var(ddof=0)
        out["graph_correlation"] = (
            out["log_return"] * out["visible_return_mean"]
            / out["return_variance"].where(out["return_variance"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["entry_threshold"] = (
            out["graph_correlation"].shift(1)
            .rolling(params.threshold_window).quantile(params.entry_percentile)
        )
        out["entry"] = (
            (out["log_return"] > 0)
            & (out["visible_return_mean"] > 0)
            & (out["graph_correlation"] > out["entry_threshold"])
            & np.isfinite(out["atr"]) & (out["atr"] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: VisibilityParams
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        stop = np.nan
        entry_bar = 0
        for i in range(len(data)):
            if pending:
                # The prior close's order fills at this open; ATR is frozen at that decision.
                active = True
                pending = False
                entry_bar = i
                stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                if not np.isfinite(closes[i]) or not np.isfinite(stop):
                    active = False
                elif closes[i] < stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                decisions[i] = int(active)
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = True
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": [100., 102., 104., 105., 105., 105.],
         "close": [101., 103., 110., 99., 97., 100.]}, index=index
    )
    data["high"] = data[["open", "close"]].max(axis=1) + 1
    data["low"] = data[["open", "close"]].min(axis=1) - 1
    data["volume"] = 1000.
    ind = pd.DataFrame({"entry": [True, False, False, False, False, False], "atr": 2.}, index=index)
    result = GeneratedStrategy.generate_signals(data, ind, None, VisibilityParams(stop_atr=2.0))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    result = GeneratedStrategy.generate_signals(data, ind, None, VisibilityParams(hold_bars=2))
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    real = GeneratedStrategy.indicators(data, VisibilityParams())
    assert not real["entry"].any()


if __name__ == "__main__":
    _self_check()
