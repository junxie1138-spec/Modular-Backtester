from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 40
    atr_window: int = 14
    shock_memory: int = 5
    shock_atr: float = 3.0
    release_atr: float = 1.8
    profit_target: float = 0.025
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791273579"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(200, params.peak_window + params.shock_memory + 1,
                       params.atr_window + params.shock_memory + 2))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.peak_window, params.atr_window, params.shock_memory, params.max_hold) < 1:
            raise ValueError("Window lengths and max_hold must be positive")
        if not (0 < params.release_atr < params.shock_atr) or params.profit_target <= 0:
            raise ValueError("Require 0 < release_atr < shock_atr and a positive target")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown_atr"] = ((out["peak"] - close) / out["atr"].where(out["atr"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["previous_drawdown_atr"] = out["drawdown_atr"].shift(1)
        out["recent_shock"] = out["previous_drawdown_atr"].rolling(params.shock_memory, min_periods=params.shock_memory).max()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["entry_setup"] = (
            (close > out["ma200"])
            & (close > out["previous_close"])
            & (out["atr"] > out["previous_atr"])
            & (out["recent_shock"] >= params.shock_atr)
            & (out["previous_drawdown_atr"] >= params.release_atr)
            & (out["drawdown_atr"] < params.release_atr)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        setups = indicators["entry_setup"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                # The previous close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                target_hit = valid_entry and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_hit = i - entry_bar + 1 >= params.max_hold
                if not valid_entry or target_hit or time_hit:
                    active = False
            elif setups[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
            raw[i] = int(active)
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 99.0,
                         "close": [100.0, 100.0, 100.0, 100.0, 103.0, 100.0],
                         "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry_setup": [True, False, False, True, False, False]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None,
                                       RecoveryParams(max_hold=2, profit_target=0.02))
    # First trade times out; second reaches its target on its entry session.
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 1, 0]
    assert (result.data["size"] > 0).all()
    assert strategy.warmup_bars(RecoveryParams()) >= 200


if __name__ == "__main__":
    _self_check()
