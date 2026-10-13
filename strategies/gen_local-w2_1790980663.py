from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTransferParams:
    lookback: int = 6
    gap_multiple: float = 1.4
    gap_floor: float = 0.002
    regime_threshold: float = 0.25
    profit_target: float = 0.012
    holding_bars: int = 4

    def __post_init__(self):
        if not 2 <= self.lookback <= 8 or not 3 <= self.holding_bars <= 5:
            raise ValueError("lookback must be 2..8 and holding_bars 3..5")
        if not 0 < self.regime_threshold < 1:
            raise ValueError("regime_threshold must be between zero and one")
        if not all(np.isfinite(x) and x > 0 for x in
                   (self.gap_multiple, self.gap_floor, self.profit_target)):
            raise ValueError("gap and target parameters must be finite and positive")


class GeneratedStrategy(BaseStrategy[GapTransferParams]):
    strategy_id = "gen_local-w2_1790980663"

    @classmethod
    def params_type(cls):
        return GapTransferParams

    def warmup_bars(self, params: GapTransferParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: GapTransferParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prior = data["close"].shift(1).where(lambda x: x > 0)
        opening = data["open"].where(lambda x: x > 0)
        out["gap"] = (opening / prior - 1).replace([np.inf, -np.inf], np.nan)
        out["session"] = (data["close"] / opening - 1).replace([np.inf, -np.inf], np.nan)
        n = params.lookback
        out["cross_energy"] = (out["gap"] * out["session"]).rolling(n, min_periods=n).sum().shift(1)
        out["gap_energy"] = out["gap"].pow(2).rolling(n, min_periods=n).sum().shift(1)
        out["session_energy"] = out["session"].pow(2).rolling(n, min_periods=n).sum().shift(1)
        out["gap_noise"] = out["gap"].abs().rolling(n, min_periods=n).mean().shift(1)
        out["normalizer"] = np.sqrt(out["gap_energy"] * out["session_energy"])
        out["coupling"] = out["cross_energy"] / out["normalizer"].replace(0, np.nan)
        out["gap_ratio"] = out["gap"].abs() / out["gap_noise"].replace(0, np.nan)
        out["regime"] = np.select(
            [out["coupling"] >= params.regime_threshold,
             out["coupling"] <= -params.regime_threshold],
            [1, -1], default=0
        )
        out["direction"] = np.sign(out["gap"]).fillna(0) * out["regime"]
        eligible = (
            (out["gap"].abs() >= params.gap_floor)
            & (out["gap_ratio"] >= params.gap_multiple)
            & (out["direction"] * out["session"] > 0)
        )
        out["candidate"] = out["direction"].where(eligible, 0).fillna(0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapTransferParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        entry_price = np.nan
        first_decision = self.warmup_bars(params) - 1
        for i in range(len(data)):
            if position:
                if np.isnan(entry_price):
                    # The preceding close's decision fills at this observed open.
                    entry_price, entry_bar = opens[i], i
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                if not valid:
                    position, entry_price = 0, np.nan
                else:
                    gain = position * (closes[i] / entry_price - 1)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.holding_bars:
                        # Close-confirmed exits fill at the following open.
                        position, entry_price = 0, np.nan
            elif i >= first_decision and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), GapTransferParams()
    start = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=start + params.holding_bars + 3)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=start)
    for side in (-1, 1):
        ind = pd.DataFrame({"candidate": 0}, index=index)
        ind.iloc[start - 1, 0] = side
        timed = strategy.generate_signals(data, ind, ctx, params).data["signal"]
        assert (timed.iloc[:start] == 0).all()
        assert (timed.iloc[start:start + params.holding_bars] == side).all()
        assert timed.iloc[start + params.holding_bars] == 0
        won = data.copy()
        won.loc[index[start + 1], "close"] = 100.0 * (1 + side * 0.02)
        targeted = strategy.generate_signals(won, ind, ctx, params).data["signal"]
        assert targeted.iloc[start] == side and targeted.iloc[start + 1] == side
        assert targeted.iloc[start + 2] == 0
    assert strategy.indicators(data, params)["candidate"].eq(0).all()


if __name__ == "__main__":
    _self_check()
