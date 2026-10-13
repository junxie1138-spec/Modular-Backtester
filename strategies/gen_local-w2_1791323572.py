from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RotationParams:
    rank_window: int = 60
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[RotationParams]):
    strategy_id = "gen_local-w2_1791323572"

    @classmethod
    def params_type(cls):
        return RotationParams

    def warmup_bars(self, params):
        return max(int(params.rank_window) + 5, 21)

    def indicators(self, data, params):
        if params.rank_window < 2 or not np.isfinite(params.target_vol) or params.target_vol <= 0:
            raise ValueError("rank_window must be >= 2 and target_vol must be positive")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["energy"] = out["return"].pow(2).rolling(5).sum()
        out["down_energy"] = out["return"].clip(upper=0).pow(2).rolling(5).sum()
        out["down_share"] = out["down_energy"].div(out["energy"].replace(0, np.nan))
        out["rank"] = out["down_share"].rolling(int(params.rank_window)).rank(pct=True)
        out["momentum"] = data["close"].pct_change(5, fill_method=None)
        out["vol"] = out["return"].rolling(20).std(ddof=0) * np.sqrt(252.0)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({"signal": 0, "size": 0.01}, index=data.index)
        armed = 0
        active = False
        entry = stop = peak = allocation = 0.0
        age = 0
        protected = False
        pending = False
        entry_atr = 0.0
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.01, dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        values = indicators[["rank", "momentum", "vol", "atr"]].to_numpy(dtype=float)
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            rank, momentum, vol, atr = values[i]
            if not np.isfinite(prices[i]).all() or min(prices[i]) <= 0:
                active = pending = False
                armed = 0
                continue
            if pending:
                entry = opening
                peak = entry
                stop = entry - 2.5 * entry_atr
                protected = False
                age = 0
                pending = False
            if active:
                age += 1
                # Test the previously established stop before raising it.
                # Stop breaches become next-open exits under this contract.
                if low <= stop or age >= 10 or not np.isfinite(atr) or atr <= 0:
                    active = False
                    armed = 0
                    continue
                peak = max(peak, high)
                if peak >= entry * 1.01:
                    protected = True
                if protected:
                    stop = max(stop, entry, peak - 2.5 * atr)
                signals[i] = 1
                sizes[i] = allocation
                continue
            if not np.isfinite(values[i]).all() or vol <= 0 or atr <= 0:
                armed = 0
                continue
            armed = max(0, armed - 1)
            if rank >= 0.80:
                armed = 10
            if armed > 0 and rank <= 0.35 and momentum > 0:
                allocation = float(np.clip(params.target_vol / vol, 0.01, 0.95))
                entry_atr = atr
                active = pending = True
                armed = 0
                signals[i] = 1
                sizes[i] = allocation
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0, 100.0, 100.0, 102.0, 102.0, 102.0],
        "high": [101.0, 101.0, 102.0, 104.0, 103.0, 103.0],
        "low": [99.0, 99.0, 99.0, 101.2, 100.5, 100.0],
        "close": [100.0, 100.0, 101.0, 103.0, 102.0, 102.0],
        "volume": [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        "rank": [0.9, 0.2, 0.2, 0.2, 0.2, 0.2],
        "momentum": [0.01] * 6,
        "vol": [0.2] * 6,
        "atr": [1.0, 1.0, 0.4, 10.0, 1.0, 1.0],
    }, index=index)
    strategy = GeneratedStrategy()
    params = RotationParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0]
    assert np.isclose(result["size"].iloc[2], 0.6)
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
