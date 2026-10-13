from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationBudgetParams:
    roc_bars: int = 10
    debt_bars: int = 8
    regime_bars: int = 40
    efficiency_cutoff: float = 0.35
    release_threshold: float = 1.25
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[AccelerationBudgetParams]):
    strategy_id = "gen_local_1790897446"

    @classmethod
    def params_type(cls):
        return AccelerationBudgetParams

    def warmup_bars(self, params: AccelerationBudgetParams) -> int:
        return int(max(params.regime_bars + 1, params.roc_bars + params.debt_bars + 2))

    def indicators(self, data: pd.DataFrame, params: AccelerationBudgetParams) -> pd.DataFrame:
        if min(params.roc_bars, params.debt_bars, params.regime_bars, params.max_hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (0 < params.efficiency_cutoff < 1 and params.release_threshold > 0 and params.profit_target > 0):
            raise ValueError("Invalid efficiency cutoff, release threshold, or profit target.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["price_change"] = close.diff()
        out["roc"] = close.div(close.shift(params.roc_bars)).sub(1)
        out["acceleration"] = out["roc"].diff()
        out["path_length"] = out["price_change"].abs().rolling(params.regime_bars).sum()
        out["displacement"] = close.diff(params.regime_bars).abs()
        out["efficiency"] = out["displacement"].div(out["path_length"].replace(0, np.nan))
        # Prior deceleration is a budget that today's acceleration must clear.
        out["debt"] = (-out["acceleration"].clip(upper=0)).rolling(params.debt_bars).mean().shift(1)
        out["scale"] = out["acceleration"].abs().rolling(params.debt_bars).mean().shift(1)
        out["release"] = out["acceleration"].div((out["debt"] + out["scale"]).replace(0, np.nan))
        out = out.replace([np.inf, -np.inf], np.nan)
        out["efficient"] = out["efficiency"].ge(params.efficiency_cutoff)
        out["regime_eligible"] = (
            (out["efficient"] & out["roc"].gt(0))
            | (~out["efficient"] & out["roc"].lt(0) & out["efficiency"].notna())
        )
        out["clearance"] = (
            out["release"].gt(params.release_threshold)
            & out["release"].shift(1).le(params.release_threshold)
            & out["debt"].gt(1e-10)
        )
        ready = np.arange(len(data)) >= self.warmup_bars(params)
        out["entry"] = (out["clearance"] & out["regime_eligible"] & ready).fillna(False)
        out["conviction"] = (0.25 + 0.70 * out["release"].div(3).clip(0, 1)).fillna(0.25)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AccelerationBudgetParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.25}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].fillna(0.25).clip(0.25, 0.95).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        held_size = 0.25
        for i in range(len(data)):
            if active:
                # Read the entry open only when that bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0
                target_hit = valid and np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.max_hold_bars
                if not valid or target_hit or time_up:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = held_size
                # No same-close reentry after an exit.
                continue
            if entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                held_size = conviction[i]
                signals[i] = 1
                sizes[i] = held_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = AccelerationBudgetParams(roc_bars=2, debt_bars=2, regime_bars=2, max_hold_bars=3)
    index = pd.date_range("2020-01-01", periods=20, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    features = strategy.indicators(data, params)
    assert not features["entry"].any()
    features["entry"] = False
    features.loc[index[6], "entry"] = True
    features["conviction"] = 0.60
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0] * 7 + [1] * 3 + [0] * 10
    assert result["size"].iloc[7] == 0.60
    data.loc[index[8], "close"] = 105.0
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0] * 7 + [1] * 2 + [0] * 11
    prefix = strategy.generate_signals(data.iloc[:9], features.iloc[:9], ctx, params).data
    assert prefix.equals(result.iloc[:9])
