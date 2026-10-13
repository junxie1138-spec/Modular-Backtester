from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FridayParams:
    rank_cutoff: float = 0.20
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[FridayParams]):
    strategy_id = "gen_local-w2_1791214194"

    @classmethod
    def params_type(cls) -> type[FridayParams]:
        return FridayParams

    def warmup_bars(self, params: FridayParams) -> int:
        return 252

    def indicators(self, data: pd.DataFrame, params: FridayParams) -> pd.DataFrame:
        if not 0.0 < params.rank_cutoff < 1.0:
            raise ValueError("rank_cutoff must be between zero and one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("stop_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        out["friday"] = data.index.dayofweek == 4
        out["intraday_return"] = (data["close"] / data["open"].where(data["open"] > 0.0) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["friday_return"] = out["intraday_return"].where(out["friday"])
        out["friday_rank"] = out["friday_return"].rolling(252, min_periods=40).rank(pct=True)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry"] = (
            out["friday"]
            & out["intraday_return"].lt(0.0)
            & out["friday_rank"].le(params.rank_cutoff)
            & out["atr"].gt(0.0)
            & np.isfinite(out["atr"])
            & data["close"].gt(0.0)
        ).fillna(False)
        out.loc[out.index[:251], "entry"] = False
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: FridayParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        fill_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if active:
                if i == fill_bar:
                    # The prior decision fills at this open; freeze the prior ATR.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                if (
                    not np.isfinite(closes[i])
                    or not np.isfinite(stop)
                    or closes[i] < stop
                    or i - fill_bar + 1 >= 2
                ):
                    active = False
                else:
                    raw[i] = 1
            elif entries[i] and i + 1 < len(data):
                active = True
                fill_bar = i + 1
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=dates)
    indicators = pd.DataFrame({"atr": 1.0, "entry": [False, False, True, False, False, False]}, index=dates)
    strategy = GeneratedStrategy()
    params = FridayParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0]
    data.loc[dates[3], ["open", "high", "low", "close"]] = [110.0, 111.0, 107.0, 108.0]
    indicators.loc[dates[3], "atr"] = 100.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 0, 0]
    assert result["size"].gt(0.0).all()
    indicators["entry"] = False
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
