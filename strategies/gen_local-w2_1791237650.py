from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AsymmetricResponseParams:
    lookback: int = 6
    rebound_threshold: float = 0.15
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AsymmetricResponseParams]):
    strategy_id = "gen_local-w2_1791237650"

    @classmethod
    def params_type(cls):
        return AsymmetricResponseParams

    def warmup_bars(self, params: AsymmetricResponseParams) -> int:
        if not 4 <= params.lookback <= 8:
            raise ValueError("lookback must be between 4 and 8")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be 1 or 2")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if not np.isfinite(params.rebound_threshold) or params.rebound_threshold < 0:
            raise ValueError("rebound_threshold must be finite and nonnegative")
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: AsymmetricResponseParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        out = pd.DataFrame(index=data.index)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["previous_return"] = out["return"].shift(1)
        out["down_weight"] = (-out["previous_return"]).clip(lower=0)
        out["up_weight"] = out["previous_return"].clip(lower=0)
        valid = out["return"].notna() & out["previous_return"].notna()
        for side in ("down", "up"):
            weight = out[side + "_weight"].where(valid)
            out[side + "_cross"] = weight * out["return"]
            out[side + "_energy"] = weight.pow(2)
            out[side + "_cross_sum"] = out[side + "_cross"].rolling(params.lookback).sum()
            out[side + "_energy_sum"] = out[side + "_energy"].rolling(params.lookback).sum()
            out[side + "_count"] = weight.gt(0).astype(float).where(valid).rolling(params.lookback).sum()
            out[side + "_response"] = (out[side + "_cross_sum"] / out[side + "_energy_sum"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AsymmetricResponseParams) -> SignalFrame:
        self.warmup_bars(params)
        eligible = (
            indicators["return"].lt(0)
            & indicators["down_response"].gt(params.rebound_threshold)
            & indicators["up_response"].le(0)
            & indicators["down_count"].ge(2)
            & indicators["up_count"].ge(2)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry = np.nan
        age = 0
        for i in range(len(data)):
            if active:
                # A previous close's entry decision fills at this bar's open.
                if np.isnan(entry):
                    entry = opens[i]
                age += 1
                valid_entry = np.isfinite(entry) and entry > 0
                reached_target = valid_entry and np.isfinite(closes[i]) and closes[i] >= entry * (1.0 + params.profit_target)
                # Targets are checked at the close; exits fill next bar.
                if not valid_entry or reached_target or age >= params.max_hold_bars:
                    active = False
                    entry = np.nan
                    age = 0
                else:
                    decisions[i] = 1
            elif eligible[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": [100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0], "volume": 1000.0}, index=index)
    forced = pd.DataFrame({"return": -0.01, "down_response": 0.5, "up_response": -0.2, "down_count": 3.0, "up_count": 3.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=8)
    timed = strategy.generate_signals(data, forced, ctx, AsymmetricResponseParams(profit_target=0.5)).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0]
    targeted = strategy.generate_signals(data, forced, ctx, AsymmetricResponseParams()).data
    assert targeted["signal"].iloc[:3].tolist() == [0, 1, 0]
    prices = [100.0, 99.0, 100.0, 101.0, 100.0, 99.0, 100.0, 101.0, 100.0, 99.0, 100.0]
    sample = pd.DataFrame({"close": prices}, index=pd.date_range("2020-02-01", periods=len(prices)))
    derived = strategy.indicators(sample, AsymmetricResponseParams())
    assert derived["down_response"].first_valid_index() == sample.index[7]
    assert strategy.warmup_bars(AsymmetricResponseParams(lookback=8)) == 10
    assert timed["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
