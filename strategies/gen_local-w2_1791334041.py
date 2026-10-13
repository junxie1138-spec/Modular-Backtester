from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 10
    decay_ratio: float = 0.5
    position_ceiling: float = 0.35
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791334041"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.lookback + 2))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.lookback < 2 or not 0 < params.decay_ratio < 1:
            raise ValueError("Require lookback >= 2 and decay_ratio between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["range"] = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["prior_low"] = data["low"].shift(1).rolling(params.lookback, min_periods=params.lookback).min()
        out["prior_high"] = data["high"].shift(1).rolling(params.lookback, min_periods=params.lookback).max()
        out["envelope_width"] = (out["prior_high"] - out["prior_low"]).where(lambda s: s > 0)
        out["invasion"] = (out["prior_low"] - data["low"]).clip(lower=0) / out["range"]
        out["previous_invasion"] = out["invasion"].shift(1)
        out["position"] = (data["close"] - out["prior_low"]) / out["envelope_width"]
        out["close_location"] = (data["close"] - data["low"]) / out["range"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["setup"] = (
            (data["close"] > out["ma200"])
            & (out["previous_invasion"] >= 0.2)
            & (out["invasion"] > 0)
            & (out["invasion"] <= params.decay_ratio * out["previous_invasion"])
            & (out["position"] <= params.position_ceiling)
            & (out["close_location"] >= 0.65)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if not 1 <= params.max_hold_bars <= 2 or params.profit_target <= 0:
            raise ValueError("Require one or two holding bars and a positive profit target.")
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        raw = np.zeros(len(data), dtype=int)
        setups = indicators["setup"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # The preceding close scheduled entry at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                # Close-confirmed profit exits execute at the following open.
                target_hit = (
                    np.isfinite(entry_price)
                    and entry_price > 0
                    and closes[i] >= entry_price * (1 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or expired:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    raw[i] = 1
            elif setups[i] and i + 1 < len(data):
                raw[i] = 1
                entry_bar = i + 1
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    data = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": [100.0, 101.0, 100.0, 100.0, 100.0, 100.0],
            "volume": 1.0,
        },
        index=pd.date_range("2020-01-01", periods=6),
    )
    strategy = GeneratedStrategy()
    params = Params()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    setups = pd.DataFrame({"setup": [True, False, True, False, False, False]}, index=data.index)
    result = strategy.generate_signals(data, setups, ctx, params)
    assert result.data["signal"].tolist() == [0, 1, 0, 1, 1, 0]
    assert (result.data["size"] > 0).all()
    assert not strategy.indicators(data, params)["setup"].any()
    data.loc[data.index[1], "open"] = 102.0
    result = strategy.generate_signals(data, setups, ctx, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 0]
