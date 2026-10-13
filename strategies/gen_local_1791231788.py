from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    impulse_window: int = 3
    rank_window: int = 63
    atr_window: int = 14
    entry_percentile: float = 0.9
    stop_atr: float = 1.0
    hold_bars: int = 2

    def __post_init__(self):
        if min(self.impulse_window, self.rank_window, self.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0.0 < self.entry_percentile <= 1.0:
            raise ValueError("entry_percentile must be in (0, 1].")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")
        if self.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791231788"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.impulse_window + params.rank_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["body"] = data["close"] - data["open"]
        out["wick"] = (data["high"] - data["low"] - out["body"].abs()).clip(lower=0)
        out["gap"] = (data["open"] - data["close"].shift(1)).abs()
        out["advance"] = out["body"].rolling(params.impulse_window).sum()
        out["noise"] = (out["wick"] + out["gap"]).rolling(params.impulse_window).sum()
        out["efficiency"] = out["advance"] / out["noise"].replace(0.0, np.nan)
        out["percentile"] = out["efficiency"].rolling(params.rank_window).rank(pct=True)
        out["close_change"] = data["close"].diff()
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - data["close"].shift(1)).abs(),
             (data["low"] - data["close"].shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        ready = (
            (indicators["percentile"] >= params.entry_percentile)
            & (indicators["advance"] > 0)
            & (indicators["close_change"] > 0)
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        stop = np.nan
        frozen_atr = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # The prior close requested entry; this bar supplies its open.
                    stop = opens[i] - params.stop_atr * frozen_atr
                exit_now = (
                    not np.isfinite(closes[i])
                    or not np.isfinite(stop)
                    or closes[i] < stop
                    or i - entry_bar + 1 >= params.hold_bars
                )
                if exit_now:
                    entry_bar = -1
                    continue
                signal[i] = 1
            elif ready[i] and np.isfinite(closes[i]) and np.isfinite(atr[i]):
                frozen_atr = atr[i]
                entry_bar = i + 1
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 94.0,
         "close": [100.0, 95.0, 100.0, 100.0, 101.0, 102.0, 100.0],
         "volume": 1000.0}, index=index,
    )
    features = pd.DataFrame(
        {"percentile": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
         "advance": 1.0, "close_change": 1.0, "atr": 2.0}, index=index,
    )
    result = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 1, 1, 0]
    assert (result["size"] > 0).all()
    actual = GeneratedStrategy.indicators(None, data, params)
    assert actual.index.equals(data.index)
    assert actual["percentile"].isna().all()


if __name__ == "__main__":
    _self_check()
