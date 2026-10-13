from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    compression_bars: int = 3
    profit_target: float = 0.05


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791231651"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return max(2, int(params.compression_bars) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.compression_bars < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Parameters must be positive.")
        high, low = data["high"], data["low"]
        previous_high, previous_low = high.shift(1), low.shift(1)
        valid = np.isfinite(high) & np.isfinite(low) & high.ge(low) & low.gt(0)
        inside = (
            valid & valid.shift(1, fill_value=False)
            & high.le(previous_high) & low.ge(previous_low)
            & (high.lt(previous_high) | low.gt(previous_low))
        ).fillna(False)
        streak = inside.astype(int).groupby((~inside).cumsum()).cumsum()
        start = inside & ~inside.shift(1, fill_value=False)
        mother_high = previous_high.where(start).ffill()
        mother_low = previous_low.where(start).ffill()
        ready = streak.shift(1, fill_value=0).ge(params.compression_bars)
        close = data["close"]
        close_valid = np.isfinite(close) & close.gt(0)
        long_entry = ready & close_valid & close.gt(mother_high.shift(1))
        short_entry = ready & close_valid & close.lt(mother_low.shift(1))
        return pd.DataFrame({
            "inside": inside.astype(int),
            "streak": streak,
            "mother_high": mother_high,
            "mother_low": mother_low,
            "entry": np.select([long_entry, short_entry], [1, -1], default=0).astype(int),
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive.")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        if not (np.isfinite(opens).all() and np.isfinite(closes).all() and (opens > 0).all() and (closes > 0).all()):
            raise ValueError("Open and close prices must be finite and positive.")
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        decision_bar = -1
        entry_price = 0.0
        for i in range(len(data)):
            if position:
                # The previous close's entry fills at this bar's open.
                if i == decision_bar + 1:
                    entry_price = opens[i]
                gain = position * (closes[i] / entry_price - 1.0)
                held_bars = i - decision_bar
                if gain >= params.profit_target or held_bars >= 18:
                    position = 0
                    signal[i] = 0
                    continue
            elif entries[i] in (-1, 1):
                position = int(entries[i])
                decision_bar = i
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=24)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": [1] + [0] * 23}, index=index)
    result = GeneratedStrategy.generate_signals(data, entries, None, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19:].eq(0).all()
    data.loc[index[0], "close"] = 120.0
    data.loc[index[2], "close"] = 106.0
    result = GeneratedStrategy.generate_signals(data, entries, None, params).data
    assert result["signal"].iloc[:4].tolist() == [0, 1, 1, 0]
    entries.loc[index[0], "entry"] = -1
    data.loc[index[2], "close"] = 94.0
    result = GeneratedStrategy.generate_signals(data, entries, None, params).data
    assert result["signal"].iloc[:4].tolist() == [0, -1, -1, 0]
    nested = data.iloc[:5].copy()
    nested["high"] = [110.0, 108.0, 106.0, 104.0, 112.0]
    nested["low"] = [90.0, 92.0, 94.0, 96.0, 100.0]
    nested["close"] = [100.0, 100.0, 100.0, 100.0, 111.0]
    derived = GeneratedStrategy.indicators(nested, params)
    assert derived["streak"].tolist() == [0, 1, 2, 3, 0]
    assert derived["entry"].tolist() == [0, 0, 0, 0, 1]


if __name__ == "__main__":
    _self_check()
