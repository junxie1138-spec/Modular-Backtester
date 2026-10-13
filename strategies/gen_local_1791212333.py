from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 3
    volatility_bars: int = 10
    history_bars: int = 60
    acceleration_quantile: float = 0.80
    regime_quantile: float = 0.65
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791212333"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.history_bars + max(params.roc_bars + 2, params.volatility_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["volatility"] = out["return"].rolling(params.volatility_bars).std()
        out["acceleration_threshold"] = out["acceleration"].shift(1).rolling(params.history_bars).quantile(params.acceleration_quantile)
        out["regime_threshold"] = out["volatility"].shift(1).rolling(params.history_bars).quantile(params.regime_quantile)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out["stress"] = out["volatility"] > out["regime_threshold"]
        continuation = ~out["stress"] & (out["roc"] > 0)
        rebound = out["stress"] & (out["roc"] < 0) & (out["return"] > 0)
        out["entry"] = (ready & (out["acceleration"] > out["acceleration_threshold"]) & (out["acceleration"] > 0) & (continuation | rebound)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
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
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                entry_bar = i
                entry_price = opens[i]
            if entry_bar >= 0:
                # Entry fills at today's open; exits decided at close fill next open.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_expired:
                    entry_bar = -1
                else:
                    decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False]}, index=index)
    params = Params()
    timed = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 101.0
    targeted = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert targeted["signal"].tolist() == [0, 1, 0, 0, 0]
    assert (targeted["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
