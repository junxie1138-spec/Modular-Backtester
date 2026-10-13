from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 6
    volume_multiple: float = 1.3
    profit_target: float = 0.012
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791261485"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if not 2 <= params.lookback <= 8:
            raise ValueError("lookback must be between 2 and 8")
        if not 3 <= params.max_hold <= 5:
            raise ValueError("max_hold must be between 3 and 5")
        if not np.isfinite(params.volume_multiple) or params.volume_multiple <= 1:
            raise ValueError("volume_multiple must be finite and exceed 1")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        return params.lookback + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"]
        change = close.diff()
        drift = change.rolling(params.lookback, min_periods=params.lookback).sum().shift(1)
        travel = change.abs().rolling(params.lookback, min_periods=params.lookback).sum().shift(1)
        volume_base = data["volume"].shift(1).rolling(params.lookback, min_periods=params.lookback).mean()
        efficiency = drift.abs().div(travel.replace(0, np.nan))
        volume_ratio = data["volume"].div(volume_base.replace(0, np.nan))
        midpoint = (data["high"] + data["low"]) / 2
        valid = np.isfinite(data[["open", "high", "low", "close", "volume"]]).all(axis=1)
        valid &= data["open"].gt(0) & close.gt(0) & data["volume"].ge(0)
        valid &= data["high"].ge(data[["open", "close"]].max(axis=1))
        valid &= data["low"].le(data[["open", "close"]].min(axis=1))
        ready = valid & drift.notna() & travel.gt(0) & volume_base.gt(0)
        orderly = efficiency.ge(0.45) & drift.gt(0)
        choppy = efficiency.le(0.25)
        continuation = orderly & change.gt(0) & close.gt(data["open"])
        absorption = choppy & change.lt(0) & close.gt(midpoint)
        entry = ready & volume_ratio.ge(params.volume_multiple) & (continuation | absorption)
        return pd.DataFrame({
            "change": change.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "drift": drift.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "travel": travel.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "volume_base": volume_base.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "efficiency": efficiency.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "volume_ratio": volume_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "midpoint": midpoint.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "valid": valid,
            "ready": ready,
            "orderly": orderly.fillna(False),
            "choppy": choppy.fillna(False),
            "continuation": continuation.fillna(False),
            "absorption": absorption.fillna(False),
            "entry": entry.fillna(False),
        }, index=data.index)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        pending = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_bar = i
                    entry_price = opens[i]
            if entry_bar >= 0:
                # Entry uses the next open; close-observed exits fill one bar later.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                expired = i - entry_bar + 1 >= params.max_hold
                if target_hit or expired:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    decisions[i] = 1
            elif entries[i] and i + 1 < len(data):
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({"entry": [True] + [False] * 7}, index=index)
    params = Params()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[2], "close"] = 102.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
