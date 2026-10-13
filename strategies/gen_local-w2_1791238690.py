from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_span: int = 3
    history: int = 63
    compression_percentile: float = 0.30
    innovation_percentile: float = 0.80
    profit_target: float = 0.006
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791238690"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 2 * params.slope_span + params.history + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.slope_span < 1 or params.history < 2:
            raise ValueError("Lookbacks must be positive; history must be at least two.")
        if not (0 < params.compression_percentile < 1 and 0 < params.innovation_percentile < 1):
            raise ValueError("Percentiles must lie strictly between zero and one.")
        if params.profit_target <= 0 or params.hold_bars not in (1, 2):
            raise ValueError("Use a positive target and a one- or two-session holding period.")
        n = params.slope_span
        out = pd.DataFrame(index=data.index)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["log_midpoint"] = np.log(out["midpoint"].where(out["midpoint"] > 0))
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["log_range"] = np.log(data["high"].where(data["high"] > 0) / data["low"].where(data["low"] > 0))
        out["range_mean"] = out["log_range"].rolling(n).mean()
        out["midpoint_mean"] = out["log_midpoint"].rolling(n).mean()
        out["close_mean"] = out["log_close"].rolling(n).mean()
        out["midpoint_slope"] = (out["midpoint_mean"] - out["midpoint_mean"].shift(n)) / n
        out["close_slope"] = (out["close_mean"] - out["close_mean"].shift(n)) / n
        out["innovation"] = (out["midpoint_slope"] - out["close_slope"].shift(1)) / out["range_mean"].where(out["range_mean"] > 0)
        out["innovation_magnitude"] = out["innovation"].abs()
        out["compression_cutoff"] = out["range_mean"].rolling(params.history).quantile(params.compression_percentile).shift(1)
        out["innovation_cutoff"] = out["innovation_magnitude"].rolling(params.history).quantile(params.innovation_percentile).shift(1)
        out["closing_lag"] = out["log_close"] - out["log_midpoint"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        required = ["range_mean", "compression_cutoff", "innovation",
                    "innovation_cutoff", "midpoint_slope", "closing_lag"]
        ready = indicators[required].notna().all(axis=1)
        eligible = (
            ready
            & (indicators["range_mean"] <= indicators["compression_cutoff"])
            & (indicators["innovation"].abs() > indicators["innovation_cutoff"])
            & (indicators["innovation"] * indicators["midpoint_slope"] > 0)
            & (indicators["innovation"] * indicators["closing_lag"] < 0)
        )
        entries = np.where(eligible, np.sign(indicators["innovation"].fillna(0)), 0).astype(int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if position:
                # Yesterday's entry decision fills at today's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                        entry_price = np.nan
                        continue
                age += 1
                gain = position * (closes[i] / entry_price - 1.0)
                # Targets are observed at the close; exit fills next open.
                if gain >= params.profit_target or age >= params.hold_bars:
                    position = 0
                    entry_price = np.nan
                    age = 0
                    continue
            elif entries[i]:
                position = int(entries[i])
                entry_price = np.nan
                age = 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=4)
    for side in (-1, 1):
        features = pd.DataFrame({
            "range_mean": [0.01] * 4,
            "compression_cutoff": [0.02] * 4,
            "innovation": [side, 0, 0, 0],
            "innovation_cutoff": [0.5] * 4,
            "midpoint_slope": [side] * 4,
            "closing_lag": [-side] * 4,
        }, index=index)
        for target_hit in (False, True):
            close = [100.0] + [100.0 + side if target_hit else 100.0] * 3
            data = pd.DataFrame({"open": 100.0, "high": 102.0,
                                 "low": 98.0, "close": close,
                                 "volume": 1000.0}, index=index)
            result = strategy.generate_signals(data, features, None, params).data
            expected = [0, side, 0, 0] if target_hit else [0, side, side, 0]
            assert result["signal"].tolist() == expected
            assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
