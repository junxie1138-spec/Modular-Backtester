from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    pattern_window: int = 12
    percentile_window: int = 126
    entry_quantile: float = 0.75
    profit_target: float = 0.008
    max_hold: int = 2
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791325897"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.pattern_window + params.percentile_window + 3)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.pattern_window < 3 or params.percentile_window < 2:
            raise ValueError("Lookbacks must be at least 3 and 2.")
        if not 0.0 < params.entry_quantile < 1.0:
            raise ValueError("entry_quantile must be between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["intraday_return"] = (data["close"] / data["open"].where(data["open"] > 0) - 1).replace([np.inf, -np.inf], np.nan)
        a = out["intraday_return"].shift(2)
        b = out["intraday_return"].shift(1)
        c = out["intraday_return"]
        valid = a.notna() & b.notna() & c.notna()
        # Pairwise comparisons encode six orderings; ties favor earlier observations.
        out["pattern"] = ((a > b).astype(int) * 4 + (a > c).astype(int) * 2 + (b > c).astype(int)).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            frequency = (out["pattern"] == code).astype(float).where(valid).rolling(params.pattern_window).mean()
            out[f"frequency_{code}"] = frequency
            entropy = entropy - frequency * np.log(frequency.where(frequency > 0, 1.0))
        out["complexity"] = entropy / np.log(6.0)
        out["threshold"] = out["complexity"].shift(1).rolling(params.percentile_window).quantile(params.entry_quantile)
        day = data.index.day
        out["month_turn"] = (day >= 27) | (day <= 3)
        out["entry"] = out["month_turn"] & out["complexity"].gt(out["threshold"])
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        if params.max_hold not in (1, 2) or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Use a positive profit target and a one- or two-bar hold.")
        if not np.isfinite(params.size) or params.size <= 0:
            raise ValueError("size must be positive and finite.")
        raw = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # Observe the actual entry open only when its bar arrives.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    entry_bar = -1
                    continue
                held = i - entry_bar + 1
                # Close-based target: exit fills at the following open.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or held >= params.max_hold:
                    entry_bar = -1
                else:
                    raw[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                entry_bar = i + 1
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": [100.0, 101.0, 100.0, 100.0, 100.0, 100.0], "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": True}, index=index)
    strategy = GeneratedStrategy()
    params = GeneratedParams(profit_target=0.005)
    signals = strategy.generate_signals(data, entries, None, params).data
    assert signals["signal"].tolist() == [0, 1, 0, 1, 1, 0]
    assert signals["size"].gt(0).all()
    empty = strategy.generate_signals(data.iloc[:0], entries.iloc[:0], None, params).data
    assert empty.empty


if __name__ == "__main__":
    _self_check()
