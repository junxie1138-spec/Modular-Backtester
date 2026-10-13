from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CongestionParams:
    lookback: int = 40
    price_ceiling: float = 0.35
    density_floor: float = 0.70
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[CongestionParams]):
    strategy_id = "gen_local-w2_1791252654"

    @classmethod
    def params_type(cls):
        return CongestionParams

    def warmup_bars(self, params: CongestionParams) -> int:
        return int(params.lookback) + 1

    def indicators(self, data: pd.DataFrame, params: CongestionParams) -> pd.DataFrame:
        if params.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not 0.0 < params.price_ceiling <= 1.0:
            raise ValueError("price_ceiling must be in (0, 1]")
        if not 0.0 < params.density_floor <= 1.0:
            raise ValueError("density_floor must be in (0, 1]")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["fractional_range"] = ((data["high"] - data["low"]) / close).where(data["high"] > data["low"])
        out["volume_density"] = (data["volume"].where(data["volume"] > 0) / out["fractional_range"]).replace([np.inf, -np.inf], np.nan)
        out["price_rank"] = close.rolling(params.lookback, min_periods=params.lookback).rank(pct=True)
        out["density_rank"] = out["volume_density"].rolling(params.lookback, min_periods=params.lookback).rank(pct=True)
        out["eligible"] = ((out["price_rank"] <= params.price_ceiling) & (out["density_rank"] >= params.density_floor)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CongestionParams) -> SignalFrame:
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be 1 or 2")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        eligible = indicators["eligible"].fillna(False).astype(bool)
        confirmed = (eligible & eligible.shift(1, fill_value=False)).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        held_bars = 0
        for i in range(len(data)):
            if active:
                # The prior close's entry decision fills at this bar's open.
                if held_bars == 0:
                    entry_price = opens[i]
                held_bars += 1
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                target_hit = valid and closes[i] >= entry_price * (1.0 + params.profit_target)
                # Close-observed exits execute at the following open.
                if not valid or target_hit or held_bars >= params.max_hold_bars:
                    active = False
                    held_bars = 0
                    entry_price = np.nan
            elif confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                held_bars = 0
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"eligible": [True, True, False, False, False, False]}, index=index)
    params = CongestionParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    data.loc[index[2], "close"] = 101.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
