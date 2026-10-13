from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    depth_atr: float = 1.5
    stop_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1790940854"

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    @staticmethod
    def warmup_bars(params: DrawdownParams) -> int:
        return 31

    @staticmethod
    def indicators(data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if not (np.isfinite(params.depth_atr) and params.depth_atr > 0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Parameters must be finite and positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        ranges = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1)
        out["true_range"] = ranges.max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["peak"] = data["close"].rolling(21, min_periods=21).max()
        out["depth"] = out["peak"] - data["close"]
        out["depth_atr"] = out["depth"] / out["atr"].where(out["atr"] > 0)
        out["depth_change"] = out["depth"].diff()
        out["previous_change"] = out["depth_change"].shift(1)
        valid_changes = out["depth_change"].notna() & out["previous_change"].notna()
        out["reversal"] = (
            out["depth_change"] * out["previous_change"] < 0
        ).astype(float).where(valid_changes)
        out["reversal_count"] = out["reversal"].rolling(8, min_periods=8).sum()
        ready = np.isfinite(out[["atr", "depth_atr", "reversal_count"]]).all(axis=1)
        deep = out["depth_atr"] >= params.depth_atr
        rebound = (out["depth_change"] < 0) & (out["previous_change"] > 0)
        deepening = (out["depth_change"] > 0) & (out["previous_change"] > 0)
        buy = ready & deep & rebound & (out["reversal_count"] >= 4)
        sell = ready & deep & deepening & (out["reversal_count"] <= 2)
        out["candidate"] = np.select([buy, sell], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownParams,
    ) -> SignalFrame:
        if not (np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("stop_atr must be finite and positive")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        pending_entry = False
        held_bars = 0
        entry_price = 0.0
        entry_atr = 0.0
        for i in range(len(data)):
            if position:
                # Yesterday's entry decision fills at today's open.
                if pending_entry:
                    entry_price = opens[i]
                    pending_entry = False
                held_bars += 1
                valid_price = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i]) and closes[i] > 0
                )
                # Frozen entry ATR and entry price: never a trailing stop.
                stopped = valid_price and (
                    (position == 1 and closes[i] < entry_price - params.stop_atr * entry_atr)
                    or (position == -1 and closes[i] > entry_price + params.stop_atr * entry_atr)
                )
                if not valid_price or stopped or held_bars >= 2:
                    position = 0
                    held_bars = 0
                # Exit decisions remain flat for this bar; no immediate re-entry.
            elif candidates[i] in (-1, 1) and np.isfinite(atr[i]) and atr[i] > 0:
                if np.isfinite(closes[i]) and closes[i] > 0:
                    position = int(candidates[i])
                    entry_atr = atr[i]
                    held_bars = 0
                    pending_entry = True
            targets[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = targets
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100.0] * 8,
        "high": [104.0] * 8,
        "low": [97.0] * 8,
        "close": [100.0, 98.0, 100.0, 102.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 8,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0] * 8,
        "candidate": [1, 0, -1, 0, 1, 0, 0, 0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, DrawdownParams())
    # Both fixed stops, two-session expiry, and exactly one-bar execution lag.
    assert result.data["signal"].tolist() == [0, 1, 0, -1, 0, 1, 1, 0]
    assert (result.data["size"] > 0).all()
    warmup = GeneratedStrategy.indicators(data, DrawdownParams())
    assert warmup["candidate"].eq(0).all()


if __name__ == "__main__":
    _self_check()
