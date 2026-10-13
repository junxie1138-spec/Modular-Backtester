from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRejectionParams:
    gap_lookback: int = 20
    gap_multiple: float = 1.8
    wick_threshold: float = 0.45
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapRejectionParams]):
    strategy_id = "gen_local_1790880250"

    @classmethod
    def params_type(cls):
        return GapRejectionParams

    def warmup_bars(self, params: GapRejectionParams) -> int:
        return int(params.gap_lookback) + 2

    def indicators(self, data: pd.DataFrame, params: GapRejectionParams) -> pd.DataFrame:
        if params.gap_lookback < 2 or params.gap_multiple <= 0:
            raise ValueError("Gap lookback must be at least two and gap multiple positive.")
        if not 0 < params.wick_threshold <= 1:
            raise ValueError("Wick threshold must be in (0, 1].")
        if params.profit_target <= 0 or params.hold_bars < 1:
            raise ValueError("Profit target and holding period must be positive.")
        out = pd.DataFrame(index=data.index)
        out["gap"] = data["open"].div(data["close"].shift(1).where(data["close"].shift(1) > 0)) - 1.0
        out["gap_scale"] = out["gap"].abs().rolling(params.gap_lookback, min_periods=params.gap_lookback).mean().shift(1)
        out["range"] = data["high"] - data["low"]
        out["lower_wick"] = data[["open", "close"]].min(axis=1) - data["low"]
        out["upper_wick"] = data["high"] - data[["open", "close"]].max(axis=1)
        out["wick_rejection"] = (out["lower_wick"] - out["upper_wick"]).div(out["range"].where(out["range"] > 0))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapRejectionParams) -> SignalFrame:
        gap_agrees = indicators["gap"].lt(-params.gap_multiple * indicators["gap_scale"]) & indicators["gap_scale"].gt(0)
        wick_agrees = indicators["wick_rejection"].ge(params.wick_threshold)
        valid = np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1) & data["close"].gt(0)
        eligible = (gap_agrees & wick_agrees & valid).fillna(False).to_numpy(dtype=bool)
        eligible[:self.warmup_bars(params)] = False
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        entry_price = 0.0
        entry_bar = -1
        active = False
        for i in range(len(data)):
            if active:
                if entry_bar < 0:
                    # Yesterday's entry decision fills at today's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                # Both exits are close decisions, executed at the next open.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_expired = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_expired:
                    active = False
                    entry_bar = -1
                else:
                    raw[i] = 1
            elif eligible[i]:
                active = True
                entry_bar = -1
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapRejectionParams(gap_lookback=2)
    index = pd.date_range("2020-01-01", periods=15)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.iloc[6, data.columns.get_loc("close")] = 101.3
    indicators = pd.DataFrame({"gap": 0.0, "gap_scale": 0.01, "wick_rejection": 0.0}, index=index)
    indicators.loc[index[[4, 8]], "gap"] = -0.03
    indicators.loc[index[[4, 8]], "wick_rejection"] = 0.6
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=4)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    indicators["wick_rejection"] = 0.0
    assert not strategy.generate_signals(data, indicators, ctx, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
