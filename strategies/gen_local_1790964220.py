from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapBreakoutParams:
    gap_atr: float = 0.35
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[GapBreakoutParams]):
    strategy_id = "gen_local_1790964220"

    @classmethod
    def params_type(cls) -> type[GapBreakoutParams]:
        return GapBreakoutParams

    def warmup_bars(self, params: GapBreakoutParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: GapBreakoutParams) -> pd.DataFrame:
        if not (np.isfinite(params.gap_atr) and params.gap_atr > 0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Parameters must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["gap"] = data["open"] - out["prior_close"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["gap_score"] = out["gap"] / out["prior_atr"].where(out["prior_atr"] > 0)
        out["gap_queue"] = out["gap"].shift(1).rolling(5, min_periods=5).sum()
        out["upper"] = data["high"].rolling(20, min_periods=20).max().shift(1)
        out["lower"] = data["low"].rolling(20, min_periods=20).min().shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1) & (out["atr"] > 0)
        long_entry = (
            ready & (out["gap_score"] >= params.gap_atr)
            & (out["gap_queue"] < 0)
            & (data["open"] > out["upper"]) & (data["close"] > out["upper"])
        )
        short_entry = (
            ready & (out["gap_score"] <= -params.gap_atr)
            & (out["gap_queue"] > 0)
            & (data["open"] < out["lower"]) & (data["close"] < out["lower"])
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapBreakoutParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = 0.0
        entry_atr = 0.0
        for i in range(len(data)):
            if side:
                # The preceding close's decision fills at this bar's open.
                if entry_bar == -1:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                # Freeze decision-bar ATR; the stop never follows later prices.
                stopped = (
                    not np.isfinite(closes[i]) or closes[i] <= 0
                    or side * (closes[i] - entry_price) < -params.stop_atr * entry_atr
                )
                if stopped or i - entry_bar + 1 >= 18:
                    side = 0
                    entry_bar = -1
                signals[i] = side
                continue
            if entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(entries[i])
                entry_atr = atr[i]
                entry_bar = -1
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=40)
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 96.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    data.loc[index[3], "close"] = 97.0
    data.loc[index[8], "close"] = 103.0
    indicators = pd.DataFrame({"entry": 0, "atr": 1.0}, index=index)
    indicators.loc[index[[1, 6, 11]], "entry"] = [1, -1, 1]
    strategy = GeneratedStrategy()
    params = GapBreakoutParams(stop_atr=2.0)
    result = strategy.generate_signals(data, indicators, None, params).data
    expected = np.zeros(40, dtype=int)
    expected[2:4] = 1
    expected[7:9] = -1
    expected[12:30] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result.index.equals(data.index) and result["size"].gt(0).all()
    warmup = strategy.indicators(data, params)
    assert warmup["entry"].iloc[:20].eq(0).all()


if __name__ == "__main__":
    _self_check()
