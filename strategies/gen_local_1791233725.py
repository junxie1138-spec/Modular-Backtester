from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeShiftParams:
    change_threshold: float = 2.0
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[RangeShiftParams]):
    strategy_id = "gen_local_1791233725"

    @classmethod
    def params_type(cls) -> type[RangeShiftParams]:
        return RangeShiftParams

    def warmup_bars(self, params: RangeShiftParams) -> int:
        return 24

    def indicators(self, data: pd.DataFrame, params: RangeShiftParams) -> pd.DataFrame:
        if not np.isfinite(params.change_threshold) or params.change_threshold <= 0:
            raise ValueError("change_threshold must be positive and finite")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["midpoint_delta"] = out["midpoint"].diff()
        out["recent_range"] = out["range"].rolling(3, min_periods=3).mean()
        out["baseline_range"] = out["range"].rolling(20, min_periods=20).mean().shift(3)
        out["baseline_std"] = out["range"].rolling(20, min_periods=20).std(ddof=1).shift(3)
        out["change_score"] = (out["recent_range"] - out["baseline_range"]) / (out["baseline_std"].replace(0.0, np.nan) * np.sqrt(1.0 / 3.0 + 1.0 / 20.0))
        out["migration"] = out["midpoint_delta"].rolling(3, min_periods=3).sum()
        out["migration_path"] = out["midpoint_delta"].abs().rolling(3, min_periods=3).sum()
        out["coherence"] = out["migration"].abs() / out["migration_path"].replace(0.0, np.nan)
        out["close_location"] = (data["close"] - data["low"]) / out["range"].replace(0.0, np.nan)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([out["range"], (data["high"] - out["previous_close"]).abs(), (data["low"] - out["previous_close"]).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        changed = (out["change_score"] > params.change_threshold) & (out["change_score"].shift(1) <= params.change_threshold)
        eligible = changed & (out["coherence"] >= 0.65) & (out["atr"] > 0.0)
        long_entry = eligible & (out["migration"] > 0.0) & (out["close_location"] >= 0.65)
        short_entry = eligible & (out["migration"] < 0.0) & (out["close_location"] <= 0.35)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeShiftParams) -> SignalFrame:
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        stop_distance = np.nan
        fixed_stop = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    if np.isfinite(opens[i]) and opens[i] > 0.0:
                        fixed_stop = opens[i] - position * stop_distance
                    else:
                        position = 0
                if position:
                    stopped = np.isfinite(closes[i]) and position * (closes[i] - fixed_stop) < 0.0
                    expired = i - entry_bar + 1 >= 4
                    if stopped or expired:
                        position = 0
            elif entries[i] in (-1, 1) and np.isfinite(atrs[i]) and atrs[i] > 0.0:
                position = int(entries[i])
                entry_bar = i + 1
                stop_distance = params.stop_atr * atrs[i]
                fixed_stop = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9, freq="D")
    data = pd.DataFrame({"open": [100.0, 102.0] + [100.0] * 7, "high": 104.0, "low": 98.0, "close": [100.0, 101.0, 99.0] + [100.0] * 6, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [1, 0, 0, -1, 0, 0, 0, 0, 0], "atr": 1.0}, index=index)
    strategy = GeneratedStrategy()
    params = RangeShiftParams(stop_atr=2.0)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, -1, -1, 0]
    data.loc[index[5], "close"] = 103.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 0, 0]
    assert (result["size"] > 0.0).all()
    assert strategy.indicators(data, params)["entry"].eq(0).all()


if __name__ == "__main__":
    _self_check()
