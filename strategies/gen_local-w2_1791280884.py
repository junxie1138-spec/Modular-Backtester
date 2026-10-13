from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRefillParams:
    channel_bars: int = 20
    min_gap_pressure: float = 0.35
    profit_target: float = 0.015
    hold_bars: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GapRefillParams]):
    strategy_id = "gen_local-w2_1791280884"

    @classmethod
    def params_type(cls):
        return GapRefillParams

    def warmup_bars(self, params: GapRefillParams) -> int:
        return max(2, int(params.channel_bars)) + 1

    def indicators(self, data: pd.DataFrame, params: GapRefillParams) -> pd.DataFrame:
        n = max(2, int(params.channel_bars))
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["floor"] = data["low"].shift(1).rolling(n, min_periods=n).min()
        out["ceiling"] = data["high"].shift(1).rolling(n, min_periods=n).max()
        out["range_scale"] = (
            (data["high"] - data["low"]).shift(1)
            .rolling(n, min_periods=n).mean().replace(0.0, np.nan)
        )
        out["gap_pressure"] = (out["prior_close"] - data["open"]) / out["range_scale"]
        out["overflow"] = (out["floor"] - data["open"]) / out["range_scale"]
        out["refill"] = (
            (data["close"] - data["open"])
            / (out["prior_close"] - data["open"]).replace(0.0, np.nan)
        ).clip(0.0, 1.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["strength"] = (
            out["overflow"].clip(0.0, 1.0) * out["refill"]
        ).fillna(0.0)
        out["entry"] = (
            out["gap_pressure"].ge(params.min_gap_pressure)
            & data["open"].lt(out["floor"])
            & data["close"].ge(out["floor"])
            & data["close"].le(out["floor"] + 0.4 * (out["ceiling"] - out["floor"]))
            & out["strength"].gt(0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRefillParams,
    ) -> SignalFrame:
        if not (
            0.0 < params.min_size <= params.max_size <= 1.0
            and params.profit_target > 0.0
            and params.hold_bars >= 1
        ):
            raise ValueError("Invalid sizing or exit parameters")
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, params.min_size, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strengths = indicators["strength"].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        active = False
        entry_price = 0.0
        entry_bar = 0
        held_size = params.min_size
        for i in range(count):
            # Yesterday's entry decision fills at today's open.
            if i > 0 and signals[i - 1] == 1 and not active:
                if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                    continue
                entry_price = opens[i]
                entry_bar = i
                active = True
            if active:
                # Close-confirmed exits execute at the following open.
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = held_size
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                held_size = params.min_size + (params.max_size - params.min_size) * strengths[i]
                signals[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapRefillParams()
    idx = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=idx,
    )
    ind = pd.DataFrame(
        {"entry": [True, False, False, False, False, False, False], "strength": 0.5},
        index=idx,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert np.allclose(result.loc[result["signal"].eq(1), "size"], 0.6)
    data.loc[idx[2], "close"] = 102.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert not strategy.indicators(data, params)["entry"].any()
    assert strategy.warmup_bars(params) == 21


if __name__ == "__main__":
    _self_check()
