from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    profit_target: float = 0.05
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791277145"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        # Twenty-bar volume baseline followed by ten-bar acceleration sums.
        return 30

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        volume = data["volume"].astype(float).replace([np.inf, -np.inf], np.nan)
        volume = volume.where(volume > 0)
        out["roc_5"] = close.div(close.shift(5)).sub(1)
        out["roc_20"] = close.div(close.shift(20)).sub(1)
        out["acceleration"] = out["roc_5"].diff()
        out["volume_baseline"] = volume.rolling(20, min_periods=20).median()
        out["relative_volume"] = volume.div(out["volume_baseline"])
        valid = out["acceleration"].notna() & out["relative_volume"].notna()
        quiet = out["relative_volume"] < 1.0
        out["quiet_acceleration"] = out["acceleration"].where(quiet, 0.0).where(valid)
        out["loud_acceleration"] = out["acceleration"].where(~quiet, 0.0).where(valid)
        out["quiet_work"] = out["quiet_acceleration"].rolling(10, min_periods=10).sum()
        out["loud_work"] = out["loud_acceleration"].rolling(10, min_periods=10).sum()
        out["net_work"] = out["quiet_work"] + out["loud_work"]
        out["qualified"] = (
            (out["roc_20"] > 0)
            & (out["acceleration"] > 0)
            & (out["quiet_work"] > 0)
            & (out["loud_work"] <= 0)
            & (out["net_work"] > 0)
        ).fillna(False)
        out["entry"] = out["qualified"] & ~out["qualified"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if isinstance(params.max_hold_bars, bool) or not isinstance(params.max_hold_bars, int) or params.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be a positive integer")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    # This is the current bar's open, following yesterday's decision.
                    entry_price = opens[i]
                    entry_bar = i
                    active = True
            exited = False
            if active:
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_up:
                    active = False
                    exited = True
            if not active and not exited and entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending_entry = True
            decisions[i] = int(active or pending_entry)
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 120.0, "low": 90.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"entry": [True] + [False] * 7}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=30)
    timed = strategy.generate_signals(data, indicators, ctx, StrategyParams(0.05, 3)).data
    assert timed["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    data.iloc[1, data.columns.get_loc("open")] = 110.0
    data.iloc[1, data.columns.get_loc("close")] = 110.0
    data.iloc[2, data.columns.get_loc("close")] = 116.0
    targeted = strategy.generate_signals(data, indicators, ctx, StrategyParams(0.05, 7)).data
    assert targeted["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert (targeted["size"] > 0).all()
    assert not strategy.indicators(data, StrategyParams())["entry"].any()


if __name__ == "__main__":
    _self_check()
