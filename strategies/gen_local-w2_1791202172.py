from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTideParams:
    lookback: int = 8
    overlap_cut: float = 0.12
    tail: float = 0.80
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[RangeTideParams]):
    strategy_id = "gen_local-w2_1791202172"

    @classmethod
    def params_type(cls) -> type[RangeTideParams]:
        return RangeTideParams

    def warmup_bars(self, params: RangeTideParams) -> int:
        if not 2 <= params.lookback <= 10:
            raise ValueError("lookback must be between 2 and 10")
        if not 15 <= params.holding_bars <= 20:
            raise ValueError("holding_bars must be between 15 and 20")
        if not 0.0 < params.overlap_cut <= 1.0:
            raise ValueError("overlap_cut must be in (0, 1]")
        if not 0.5 < params.tail <= 1.0:
            raise ValueError("tail must be in (0.5, 1]")
        return int(params.lookback)

    def indicators(self, data: pd.DataFrame, params: RangeTideParams) -> pd.DataFrame:
        n = self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["envelope_high"] = data["high"].rolling(n, min_periods=n).max()
        out["envelope_low"] = data["low"].rolling(n, min_periods=n).min()
        out["shared_high"] = data["high"].rolling(n, min_periods=n).min()
        out["shared_low"] = data["low"].rolling(n, min_periods=n).max()
        out["span"] = out["envelope_high"] - out["envelope_low"]
        out["overlap"] = (
            (out["shared_high"] - out["shared_low"]).clip(lower=0.0)
            / out["span"].where(out["span"] > 0.0)
        )
        out["percentile"] = data["close"].rolling(n, min_periods=n).rank(
            method="average", pct=True
        )
        extreme = np.select(
            [out["percentile"] >= params.tail,
             out["percentile"] <= 1.0 - params.tail],
            [1, -1], default=0
        )
        out["entry"] = np.where(
            out["overlap"] >= params.overlap_cut, -extreme, extreme
        )
        ready = out["overlap"].notna() & out["percentile"].notna()
        out["entry"] = out["entry"].where(ready, 0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RangeTideParams
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i == exit_bar:
                    position = 0
                    # Keep the expiry bar flat; a new entry needs a later bar.
                else:
                    signal[i] = position
                continue
            if entry != 0:
                position = int(entry)
                exit_bar = i + params.holding_bars
                signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RangeTideParams()
    index = pd.date_range("2020-01-01", periods=45, freq="B")
    close = pd.Series(100.0 + np.arange(45), index=index)
    data = pd.DataFrame({
        "open": close, "high": close + 2.0, "low": close - 2.0,
        "close": close, "volume": 1000.0
    })
    derived = strategy.indicators(data, params)
    assert (derived["entry"].iloc[:params.lookback - 1] == 0).all()
    assert derived.iloc[:30].equals(strategy.indicators(data.iloc[:30], params))
    for direction in (-1, 1):
        entries = pd.DataFrame({"entry": direction}, index=index)
        result = strategy.generate_signals(data, entries, None, params).data
        assert result["signal"].iloc[0] == 0
        assert (result["signal"].iloc[1:19] == direction).all()
        assert result["signal"].iloc[19] == 0
        assert result["signal"].iloc[20] == direction
        assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
