from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    min_drawdown: float = 0.012
    volume_repayment: float = 0.6
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791327012"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return 7

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["prior_peak"] = data["close"].rolling(6, min_periods=6).max().shift(1)
        out["prior_drawdown"] = data["close"].shift(1) / out["prior_peak"].replace(0, np.nan) - 1.0
        out["down_volume"] = data["volume"].where(out["return"].lt(0), 0.0)
        out["down_value"] = out["down_volume"] * data["close"]
        out["prior_sell_volume"] = out["down_volume"].rolling(4, min_periods=4).sum().shift(1)
        out["prior_sell_value"] = out["down_value"].rolling(4, min_periods=4).sum().shift(1)
        out["seller_price"] = out["prior_sell_value"] / out["prior_sell_volume"].replace(0, np.nan)
        out["prior_down_days"] = out["return"].lt(0).astype(int).rolling(4, min_periods=4).sum().shift(1)
        out["volume_repayment"] = data["volume"] / out["prior_sell_volume"].replace(0, np.nan)
        out["entry"] = (
            out["prior_drawdown"].le(-params.min_drawdown)
            & out["prior_down_days"].ge(2)
            & out["return"].gt(0)
            & data["close"].shift(1).le(out["seller_price"])
            & data["close"].gt(out["seller_price"])
            & out["volume_repayment"].ge(params.volume_repayment)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        if not (0 < params.min_drawdown < 1 and params.volume_repayment > 0
                and params.profit_target > 0 and 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid recovery parameters")
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # The preceding decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    entry_bar = -1
                    continue
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                timed_out = i - entry_bar + 1 >= params.hold_bars
                if target_hit or timed_out:
                    entry_bar = -1
                    continue
                signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                entry_bar = i + 1
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 89.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[0], "close"] = 90.0
    data.loc[index[2], "close"] = 101.5
    entries = pd.DataFrame({"entry": False}, index=index)
    entries.loc[index[[0, 5]], "entry"] = True
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=7)
    result = strategy.generate_signals(data, entries, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) <= 10
    assert not strategy.indicators(data.iloc[:4], params)["entry"].any()


if __name__ == "__main__":
    _self_check()
