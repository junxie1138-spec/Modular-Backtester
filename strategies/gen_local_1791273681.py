from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 4
    percentile_bars: int = 252
    tail_percentile: float = 0.90
    pair_bars: int = 5
    profit_target: float = 0.035
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791273681"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.roc_bars + params.percentile_bars + params.pair_bars + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.roc_bars, params.percentile_bars, params.pair_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not 0.5 < params.tail_percentile < 1.0 or not params.profit_target > 0:
            raise ValueError("Invalid percentile or profit target")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["acceleration"] = out["roc"].diff()
        history = out["acceleration"].shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars)
        out["lower"] = history.quantile(1.0 - params.tail_percentile)
        out["upper"] = history.quantile(params.tail_percentile)
        out["negative_tail"] = out["acceleration"].lt(out["lower"]) & out["acceleration"].lt(0)
        out["positive_tail"] = out["acceleration"].gt(out["upper"]) & out["acceleration"].gt(0)
        out["recent_negative"] = out["negative_tail"].astype(float).shift(1).rolling(params.pair_bars, min_periods=params.pair_bars).max()
        out["recent_positive"] = out["positive_tail"].astype(float).shift(1).rolling(params.pair_bars, min_periods=params.pair_bars).max()
        ready = out[["roc", "acceleration", "lower", "upper", "recent_negative", "recent_positive"]].notna().all(axis=1)
        long_entry = ready & out["positive_tail"] & out["recent_negative"].eq(1) & out["roc"].lt(0)
        short_entry = ready & out["negative_tail"] & out["recent_positive"].eq(1) & out["roc"].gt(0)
        out["setup"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # A decision on the previous close enters at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry and valid_close else np.nan
                if not valid_entry or gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                    position = 0
            elif setups[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(setups[i])
                entry_bar = i + 1
                entry_price = np.nan
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 105.0, "low": 99.0, "close": [100., 100., 104., 100., 100., 100., 100., 100., 100.], "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"setup": [1, 0, 0, -1, 0, 0, 0, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params(profit_target=0.03, hold_bars=3)).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, -1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
