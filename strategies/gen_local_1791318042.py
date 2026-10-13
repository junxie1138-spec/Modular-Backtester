from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    volume_window: int = 20
    pressure_window: int = 5
    entry_sessions: int = 3
    volume_multiple: float = 1.15
    profit_target: float = 0.02
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791318042"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        if params.volume_window < 1 or params.pressure_window < 1:
            raise ValueError("Rolling windows must be positive.")
        if not 1 <= params.entry_sessions <= 5:
            raise ValueError("entry_sessions must be between 1 and 5.")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5.")
        if not np.isfinite(params.volume_multiple) or params.volume_multiple <= 0:
            raise ValueError("volume_multiple must be finite and positive.")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive.")
        return int(max(200, params.volume_window + 1, params.pressure_window + 2))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        volume = data["volume"].where(data["volume"] >= 0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["return"] = close.pct_change(fill_method=None)
        out["dollar_volume"] = close * volume
        out["signed_flow"] = np.sign(out["return"]) * out["dollar_volume"]
        out["prior_flow"] = out["signed_flow"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum().shift(1)
        out["prior_volume"] = volume.rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean().shift(1)
        out["volume_ratio"] = volume / out["prior_volume"].where(out["prior_volume"] > 0)
        months = pd.Series(data.index.to_period("M"), index=data.index)
        out["month_session"] = months.groupby(months, sort=False).cumcount() + 1
        # A truncated first month has no reliable observed session count.
        out["calendar_ready"] = months.ne(months.iloc[0]) if len(months) else False
        out["entry"] = (
            out["calendar_ready"]
            & out["month_session"].le(params.entry_sessions)
            & close.gt(out["ma200"])
            & out["prior_flow"].lt(0)
            & out["return"].gt(0)
            & close.gt(data["open"])
            & out["volume_ratio"].ge(params.volume_multiple)
            & np.isfinite(close)
            & np.isfinite(out["volume_ratio"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        entry_price = 0.0
        age = 0
        for i in range(len(data)):
            # Yesterday's decision becomes a position at today's open.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i]
                    age = 0
                    active = True
            if active:
                age += 1
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or age >= params.hold_bars:
                    active = False
                else:
                    desired[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                desired[i] = 1
                pending = True
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2020-01-02", periods=12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.loc[index[8], "close"] = 103.0
    indicators = pd.DataFrame({"entry": False}, index=index)
    indicators.loc[index[[0, 6]], "entry"] = True
    strategy = GeneratedStrategy()
    params = Params()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= 200
    assert not strategy.indicators(data, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
