from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    expansion_ratio: float = 1.25
    contraction_ratio: float = 0.85
    gap_atr: float = 0.20
    body_atr: float = 0.15
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791308241"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not (0 < params.contraction_ratio < 1 < params.expansion_ratio):
            raise ValueError("ATR regime thresholds must straddle one.")
        if not all(np.isfinite(v) and v > 0 for v in
                   (params.gap_atr, params.body_atr, params.profit_target)):
            raise ValueError("Entry and profit thresholds must be positive and finite.")
        if not 3 <= params.hold_bars <= 5 or int(params.hold_bars) != params.hold_bars:
            raise ValueError("hold_bars must be an integer from three to five.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["fast_atr"] = out["true_range"].rolling(2, min_periods=2).mean()
        out["reference_atr"] = out["true_range"].rolling(8, min_periods=8).mean().shift(1)
        denominator = out["reference_atr"].where(out["reference_atr"] > 0)
        out["volatility_ratio"] = out["fast_atr"] / denominator
        out["gap"] = (data["open"] - out["previous_close"]) / denominator
        out["body"] = (data["close"] - data["open"]) / denominator
        out["coherence"] = out["gap"] * out["body"]
        out["ready"] = np.isfinite(out[["volatility_ratio", "gap", "body"]]).all(axis=1)
        material = out["ready"] & (out["gap"].abs() >= params.gap_atr) & (
            out["body"].abs() >= params.body_atr)
        expanding = material & (out["volatility_ratio"] >= params.expansion_ratio) & (
            out["coherence"] > 0)
        relaxing = material & (out["volatility_ratio"] <= params.contraction_ratio) & (
            out["coherence"] < 0)
        out["entry"] = np.where(expanding | relaxing, np.sign(out["body"]), 0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        entries = indicators["entry"].to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side, held = 0, 0
        entry_price = 0.0
        for i in range(len(data)):
            if side:
                # Record the next-open fill only when its bar is reached.
                if held == 0:
                    entry_price = opens[i]
                held += 1
                gain = side * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or held >= params.hold_bars:
                    side, held = 0, 0
                raw[i] = side
            elif i >= self.warmup_bars(params) - 1 and entries[i]:
                side = int(entries[i])
                held = 0
                raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=24, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 97.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    strategy, params = GeneratedStrategy(), GeneratedParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=10)
    ind = strategy.indicators(data, params)
    assert np.isfinite(ind.to_numpy(dtype=float)).all()
    assert not ind["ready"].iloc[:9].any() and ind["ready"].iloc[9]
    ind["entry"] = 0
    ind.iloc[10, ind.columns.get_loc("entry")] = 1
    signals = strategy.generate_signals(data, ind, ctx, params).data["signal"]
    assert signals.iloc[11:15].tolist() == [1, 1, 1, 1]
    assert signals.iloc[10] == 0 and signals.iloc[15] == 0
    for side in (1, -1):
        ind.iloc[10, ind.columns.get_loc("entry")] = side
        data.loc[index[12], "close"] = 100.0 * (1 + side * 0.02)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].iloc[11:14].tolist() == [side, side, 0]
        prefix = strategy.generate_signals(data.iloc[:13], ind.iloc[:13], ctx, params).data
        pd.testing.assert_frame_equal(prefix, result.iloc[:13])
    empty = data.iloc[:0]
    assert strategy.generate_signals(empty, strategy.indicators(empty, params), ctx, params).data.empty


if __name__ == "__main__":
    _self_check()
