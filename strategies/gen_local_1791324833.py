from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    position_ceiling: float = 0.4
    profit_target: float = 0.006

    def __post_init__(self):
        if not np.isfinite(self.position_ceiling) or not 0 < self.position_ceiling < 1:
            raise ValueError("position_ceiling must be between zero and one")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791324833"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 13

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["prior_low"] = data["low"].rolling(12, min_periods=12).min().shift(1)
        out["prior_high"] = data["high"].rolling(12, min_periods=12).max().shift(1)
        width = (out["prior_high"] - out["prior_low"]).where(
            out["prior_high"] > out["prior_low"]
        )
        out["position"] = (close - out["prior_low"]) / width
        out["roc"] = close.pct_change(2, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["previous_acceleration"] = out["acceleration"].shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            out["position"].between(0, params.position_ceiling)
            & (out["roc"] < 0)
            & (out["acceleration"] > 0)
            & (out["previous_acceleration"] <= 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # The previous close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                # Close-observed target; exit fills at the next open.
                if (
                    not np.isfinite(entry_price)
                    or entry_price <= 0
                    or target_hit
                    or i - entry_bar + 1 >= 2
                ):
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    desired[i] = 1
            elif entry[i]:
                desired[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1.0},
        index=index,
    )
    entries = pd.DataFrame({"entry": [True, False, False, False]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=13)
    for first_close, expected in [
        (100.0, [0, 1, 1, 0]),
        (101.0, [0, 1, 0, 0]),
    ]:
        data.loc[index[1], "close"] = first_close
        result = strategy.generate_signals(data, entries, ctx, params).data
        assert result["signal"].tolist() == expected
        assert (result["size"] > 0).all()
    assert not strategy.indicators(data, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
