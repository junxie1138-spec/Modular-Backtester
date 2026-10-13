from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    lookback: int = 20
    entry_z: float = 1.5
    range_decay: float = 0.7
    profit_target: float = 0.012
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local_1790859190"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return max(params.lookback, 4)

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        if params.lookback < 2 or not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("lookback must be >= 2 and entry_z must be positive")
        if not np.isfinite(params.range_decay) or not 0 < params.range_decay < 1:
            raise ValueError("range_decay must be between zero and one")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.lookback, min_periods=params.lookback).mean()
        out["std"] = close.rolling(params.lookback, min_periods=params.lookback).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].where(out["std"] > 0)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["prior_range"] = out["range"].rolling(3, min_periods=3).max().shift(1)
        out["range_ratio"] = out["range"] / out["prior_range"].where(out["prior_range"] > 0)
        out["prior_low"] = data["low"].rolling(3, min_periods=3).min().shift(1)
        out["prior_high"] = data["high"].rolling(3, min_periods=3).max().shift(1)
        out["close_location"] = (close - data["low"]) / out["range"].where(out["range"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        decayed = (out["range_ratio"] > 0) & (out["range_ratio"] <= params.range_decay)
        long_entry = (
            decayed & (out["z"] <= -params.entry_z)
            & (data["low"] < out["prior_low"]) & (out["close_location"] >= 0.65)
        )
        short_entry = (
            decayed & (out["z"] >= params.entry_z)
            & (data["high"] > out["prior_high"]) & (out["close_location"] <= 0.35)
        )
        out["entry_signal"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        out["entry_size"] = (
            (out["z"].abs() / params.entry_z) * (1.0 - out["range_ratio"])
        ).clip(lower=0.25, upper=1.0).fillna(0.25)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ShockwaveParams,
    ) -> SignalFrame:
        if not 3 <= params.max_hold <= 5:
            raise ValueError("max_hold must be between three and five bars")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive")
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.25, dtype=float)
        entries = indicators["entry_signal"].to_numpy()
        strength = indicators["entry_size"].to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side, entry_bar, entry_price, trade_size = 0, -1, np.nan, 0.25
        for i in range(n):
            if side:
                # Observe the next opening price only once its bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                if (
                    not np.isfinite(entry_price) or entry_price <= 0
                    or not np.isfinite(closes[i]) or closes[i] <= 0
                    or side * (closes[i] / entry_price - 1.0) >= params.profit_target
                    or i - entry_bar + 1 >= params.max_hold
                ):
                    side = 0
            elif entries[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(entries[i])
                entry_bar = i + 1
                trade_size = float(strength[i])
            signals[i] = side
            if side:
                sizes[i] = trade_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = ShockwaveParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=20)
    for side in (1, -1):
        ind = pd.DataFrame(
            {"entry_signal": [side, 0, 0, 0, 0, 0], "entry_size": [0.6] * 6},
            index=index,
        )
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, side, side, side, side, 0]
        assert result["size"].iloc[1] == 0.6
        target_data = data.copy()
        target_data.loc[index[1], "close"] = 100.0 * (1.0 + side * 0.02)
        result = strategy.generate_signals(target_data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, side, 0, 0, 0, 0]
    flat = strategy.indicators(data, params)
    assert flat["entry_signal"].eq(0).all()
    assert np.isfinite(flat["entry_size"]).all()


if __name__ == "__main__":
    _self_check()
