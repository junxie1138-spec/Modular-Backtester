from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_share: float = 0.55
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791324949"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not 0.0 < params.gap_share < 1.0 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("gap_share must be between zero and one; trail_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        opening = data["open"].where(data["open"] > 0.0)
        previous = close.shift(1)
        out["gap"] = np.log(opening / previous).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = np.log(close / opening).replace([np.inf, -np.inf], np.nan)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["gap_energy"] = out["gap"].pow(2).rolling(20, min_periods=20).sum()
        out["intraday_energy"] = out["intraday"].pow(2).rolling(20, min_periods=20).sum()
        total = out["gap_energy"] + out["intraday_energy"]
        out["gap_share"] = out["gap_energy"] / total.where(total > 0.0)
        out["gap_sum"] = out["gap"].rolling(20, min_periods=20).sum()
        out["intraday_sum"] = out["intraday"].rolling(20, min_periods=20).sum()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["volatility"] = out["return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        out["spike"] = (out["gap"].abs() > 3.0 * out["atr"].shift(1) / previous).astype(float)
        out["recent_spike"] = out["spike"].rolling(3, min_periods=3).max()
        out["entry"] = (
            (out["gap_share"] > params.gap_share)
            & (out["gap_share"].shift(1) <= params.gap_share)
            & (out["gap_sum"] > 0.0)
            & (out["intraday_sum"] < 0.0)
            & (out["gap"] > 0.0)
            & (out["recent_spike"] == 0.0)
            & (out["atr"] > 0.0)
            & out["volatility"].notna()
        ).fillna(False)
        out["target_size"] = (0.10 / out["volatility"].clip(lower=0.01)).clip(lower=0.05, upper=0.95).fillna(0.05)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry = stop = peak = 0.0
        age = 0
        cooldown_until = -1
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            valid = bool(np.isfinite(prices[i]).all() and (prices[i] > 0.0).all() and np.isfinite(atr[i]) and atr[i] > 0.0)
            if not valid:
                active = pending = False
                cooldown_until = i + 3
                continue
            if pending:
                # The previous close's decision fills at this bar's open.
                entry = opening
                peak = opening
                stop = entry - params.trail_atr * atr[i - 1]
                age = 0
                armed = False
                active = True
                pending = False
            if active:
                age += 1
                # Test the existing stop before updating it: OHLC does not reveal high/low order.
                exit_now = low <= stop or age >= 10
                if not exit_now:
                    peak = max(peak, high)
                    armed = armed or high >= entry * 1.01
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    exit_now = close <= stop
                if exit_now:
                    # Stop breaches schedule liquidation at the next open.
                    active = False
                    cooldown_until = i + 3
                else:
                    decisions[i] = 1
            elif i >= 21 and i > cooldown_until and entries[i]:
                decisions[i] = 1
                pending = True
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["target_size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    bars = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"atr": 1.0, "entry": False, "target_size": 0.5}, index=index)
    derived.loc[index[21], "entry"] = True
    bars.loc[index[22], ["high", "low", "close"]] = [102.0, 100.0, 102.0]
    bars.loc[index[23], ["open", "high", "low", "close"]] = [101.0, 102.0, 99.0, 101.0]
    result = GeneratedStrategy.generate_signals(None, bars, derived, None, Params()).data
    assert result["signal"].iloc[21:25].tolist() == [0, 1, 1, 0]
    assert set(result["signal"].unique()) <= {0, 1}
    assert result["size"].gt(0.0).all()
