from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    return_window: int = 20
    overshoot: float = 1.1
    profit_target: float = 0.006
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791265350"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        if params.return_window < 2:
            raise ValueError("return_window must be at least 2")
        if not np.isfinite(params.overshoot) or params.overshoot < 1.0:
            raise ValueError("overshoot must be finite and at least 1")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if params.max_hold not in (1, 2):
            raise ValueError("max_hold must be 1 or 2")
        # Percentage change and exclusion of today's return add two bars.
        return int(max(200, params.return_window + 2))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["prior_return"] = out["return"].shift(1)
        out["return_floor"] = out["prior_return"].rolling(params.return_window, min_periods=params.return_window).min()
        out["return_ceiling"] = out["prior_return"].rolling(params.return_window, min_periods=params.return_window).max()
        out["corridor_width"] = out["return_ceiling"] - out["return_floor"]
        out["relative_return"] = (
            (2.0 * out["return"] - out["return_floor"] - out["return_ceiling"])
            / out["corridor_width"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        long_entry = (
            (close > out["ma200"])
            & (out["return"] < 0.0)
            & (out["relative_return"] < -params.overshoot)
        )
        short_entry = (
            (close < out["ma200"])
            & (out["return"] > 0.0)
            & (out["relative_return"] > params.overshoot)
        )
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position, entry, age = 0, 0.0, 0

        for i in range(len(data)):
            if position:
                if age == 0:
                    # The previous close's decision fills at this bar's open.
                    entry = opens[i]
                age += 1
                valid = np.isfinite(entry) and entry > 0 and np.isfinite(closes[i]) and closes[i] > 0
                gain = position * (closes[i] / entry - 1.0) if valid else np.nan
                # ponytail: gross open-price proxy; use actual fills if the contract exposes them.
                if not valid or gain >= params.profit_target or age >= params.max_hold:
                    position, age = 0, 0
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                age = 0
            raw[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=4)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 98.0, "close": 100.0, "volume": 1000.0}, index=index)
    for side in (1, -1):
        indicators = pd.DataFrame({"candidate": [side, 0, 0, 0]}, index=index)
        flat = strategy.generate_signals(data, indicators, ctx, params).data
        assert flat["signal"].tolist() == [0, side, side, 0]
        assert (flat["size"] > 0).all()
        profitable = data.copy()
        profitable.loc[index[1], "close"] = 100.0 + side
        assert strategy.generate_signals(profitable, indicators, ctx, params).data["signal"].tolist() == [0, side, 0, 0]
        gap = data.copy()
        gap.loc[index[1]:, ["open", "close"]] = 102.0
        assert strategy.generate_signals(gap, indicators, ctx, params).data["signal"].tolist() == [0, side, side, 0]
        one_day = GeneratedParams(max_hold=1)
        assert strategy.generate_signals(data, indicators, ctx, one_day).data["signal"].tolist() == [0, side, 0, 0]
    flat_data = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=220))
    assert strategy.indicators(flat_data, params)["candidate"].eq(0).all()
    assert strategy.warmup_bars(GeneratedParams(return_window=250)) == 252


if __name__ == "__main__":
    _self_check()
