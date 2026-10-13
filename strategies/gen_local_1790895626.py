from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MonthlyReclaimParams:
    volume_lookback: int = 20
    volume_multiple: float = 1.05
    entry_sessions: int = 7
    profit_target: float = 0.04
    max_hold: int = 18

    def __post_init__(self):
        if self.volume_lookback < 2 or not 2 <= self.entry_sessions <= 10:
            raise ValueError("Invalid lookback or entry window")
        if self.volume_multiple <= 0 or self.profit_target <= 0 or self.max_hold < 1:
            raise ValueError("Volume multiple, target and holding limit must be positive")


class GeneratedStrategy(BaseStrategy[MonthlyReclaimParams]):
    strategy_id = "gen_local_1790895626"

    @classmethod
    def params_type(cls):
        return MonthlyReclaimParams

    def warmup_bars(self, params: MonthlyReclaimParams) -> int:
        # Two calendar months contain at most 62 daily observations.
        return int(max(64, params.volume_lookback + 1))

    def indicators(self, data: pd.DataFrame, params: MonthlyReclaimParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["month_key"] = data.index.year * 12 + data.index.month
        groups = out["month_key"]
        out["month_session"] = data["close"].groupby(groups).cumcount() + 1
        out["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3.0
        valid = (
            np.isfinite(out["typical_price"]) & (out["typical_price"] > 0)
            & np.isfinite(data["volume"]) & (data["volume"] > 0)
        )
        out["eligible_volume"] = data["volume"].where(valid, 0.0)
        out["price_volume"] = out["typical_price"].where(valid, 0.0) * out["eligible_volume"]
        out["monthly_volume"] = out["eligible_volume"].groupby(groups).cumsum()
        out["monthly_price_volume"] = out["price_volume"].groupby(groups).cumsum()
        out["monthly_vwap"] = out["monthly_price_volume"] / out["monthly_volume"].replace(0.0, np.nan)
        out["prior_month_vwap"] = out["monthly_vwap"].shift(1).where(groups.ne(groups.shift(1))).ffill()
        out["volume_baseline"] = data["volume"].shift(1).rolling(
            params.volume_lookback, min_periods=params.volume_lookback
        ).mean()
        out["volume_ratio"] = data["volume"] / out["volume_baseline"].where(out["volume_baseline"] > 0)
        out["bullish_volume"] = (
            data["close"].gt(data["close"].shift(1))
            & data["close"].gt(data["open"])
            & data["close"].gt(out["prior_month_vwap"])
            & out["volume_ratio"].ge(params.volume_multiple)
            & np.isfinite(data["close"]) & np.isfinite(data["open"])
            & data["open"].gt(0) & valid
        ).fillna(False)
        out["two_bar_confirmation"] = (
            out["bullish_volume"]
            & out["bullish_volume"].shift(1).fillna(False).astype(bool)
            & groups.eq(groups.shift(1))
        )
        out["entry"] = (
            out["two_bar_confirmation"]
            & out["month_session"].le(params.entry_sessions)
            & data["close"].shift(2).le(out["prior_month_vwap"])
            & out["prior_month_vwap"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MonthlyReclaimParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # The entry decision was yesterday; today's open is now known.
                if i == entry_bar:
                    entry_price = opens[i]
                target_hit = (
                    np.isfinite(closes[i]) and np.isfinite(entry_price)
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.max_hold
                if not np.isfinite(entry_price) or entry_price <= 0 or target_hit or expired:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signal[i] = 1
            elif i >= self.warmup_bars(params) - 1 and entries[i] and i + 1 < len(data):
                entry_bar = i + 1
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2020-01-01", periods=100)
    data = pd.DataFrame(
        {"open": 100.0, "high": 108.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.loc[index[65:67], "close"] = [103.0, 104.0]
    data.loc[index[65:67], "volume"] = 2000.0
    data.loc[index[71], "open"] = 102.0
    data.loc[index[72], "close"] = 107.0
    strategy = GeneratedStrategy()
    params = MonthlyReclaimParams(max_hold=3)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    entries = pd.DataFrame({"entry": False}, index=index)
    entries.loc[index[[70, 80]], "entry"] = True
    result = strategy.generate_signals(data, entries, ctx, params).data
    assert result["signal"].iloc[70] == 0
    assert result["signal"].iloc[71:73].eq(1).all()
    assert result["signal"].iloc[73] == 0
    assert result["signal"].iloc[81:84].eq(1).all()
    assert result["signal"].iloc[84] == 0
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
    actual = strategy.indicators(data, params)
    assert not actual["entry"].iloc[65] and actual["entry"].iloc[66]
    pd.testing.assert_frame_equal(
        strategy.indicators(data, params).iloc[:85],
        strategy.indicators(data.iloc[:85], params),
    )


if __name__ == "__main__":
    _self_check()
