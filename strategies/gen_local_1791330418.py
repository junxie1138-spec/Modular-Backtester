from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCapacityParams:
    roc_window: int = 5
    noise_window: int = 30
    acceleration_threshold: float = 0.6
    profit_target: float = 0.04
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[GapCapacityParams]):
    strategy_id = "gen_local_1791330418"

    @classmethod
    def params_type(cls) -> type[GapCapacityParams]:
        return GapCapacityParams

    def warmup_bars(self, params: GapCapacityParams) -> int:
        return int(max(2 * params.roc_window + 1, params.noise_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapCapacityParams) -> pd.DataFrame:
        if params.roc_window < 2 or params.noise_window < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not np.isfinite(params.acceleration_threshold) or params.acceleration_threshold <= 0:
            raise ValueError("Acceleration threshold must be positive and finite")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0 or params.max_hold_bars < 1:
            raise ValueError("Profit target and holding period must be positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        opening = data["open"].where(lambda s: s > 0)
        closing = data["close"].where(lambda s: s > 0)
        out["gap_log_return"] = np.log(opening / previous_close)
        out["intraday_log_return"] = np.log(closing / opening)
        out["intraday_reversal"] = (-out["intraday_log_return"]).clip(lower=0)
        out["gap_log_roc"] = out["gap_log_return"].rolling(params.roc_window).sum()
        out["gap_roc"] = np.expm1(out["gap_log_roc"])
        out["gap_acceleration"] = out["gap_roc"].diff(params.roc_window)
        out["gap_noise"] = out["gap_log_return"].rolling(params.noise_window).std(ddof=0) * np.sqrt(params.roc_window)
        out["reversal_capacity"] = out["intraday_reversal"].rolling(params.roc_window).sum()
        out["overflow"] = out["gap_log_roc"] - out["reversal_capacity"]
        out["strength"] = out["gap_acceleration"] / (out["gap_noise"].clip(lower=0.0001) * params.acceleration_threshold)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["strength"] >= 1.0)
            & (out["gap_roc"] > 0)
            & (out["overflow"] > 0)
            & (out["gap_log_return"] > 0)
            & (out["intraday_log_return"] >= 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapCapacityParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.25, dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strengths = indicators["strength"].to_numpy(dtype=float)
        openings = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_price = np.nan
        entry_bar = -1
        held_size = 0.25
        for i in range(count):
            if active:
                # A decision on the previous close fills at this session's open.
                if np.isnan(entry_price):
                    if not np.isfinite(openings[i]) or openings[i] <= 0:
                        active = False
                        continue
                    entry_price = openings[i]
                    entry_bar = i
                age = i - entry_bar + 1
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or age >= params.max_hold_bars:
                    active = False
                    entry_price = np.nan
                    sizes[i] = held_size
                    continue
            elif entries[i] and np.isfinite(strengths[i]):
                active = True
                entry_price = np.nan
                held_size = float(np.clip(0.95 * strengths[i] / (1.0 + strengths[i]), 0.25, 0.95))
            if active:
                signals[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=9, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 107.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[2], "close"] = 106.0
    indicators = pd.DataFrame({"entry": [True, False, False, False, True, False, False, False, False], "strength": [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0]}, index=index)
    strategy = GeneratedStrategy()
    params = GapCapacityParams(profit_target=0.05, max_hold_bars=3)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
    assert result["size"].iloc[5] > result["size"].iloc[1]
    assert strategy.warmup_bars(params) >= max(2 * params.roc_window + 1, params.noise_window + 1)
