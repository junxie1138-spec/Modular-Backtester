from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalEnergyParams:
    energy_share: float = 0.55
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[SeasonalEnergyParams]):
    strategy_id = "gen_local-w2_1791201039"

    @classmethod
    def params_type(cls):
        return SeasonalEnergyParams

    def warmup_bars(self, params):
        return 525

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["log_return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        for half in (1, 2, 4, 8):
            mean = out["log_return"].rolling(half, min_periods=half).mean()
            out[f"haar_{2 * half}"] = np.sqrt(half / 2.0) * (mean - mean.shift(half))
            out[f"energy_{2 * half}"] = out[f"haar_{2 * half}"].pow(2).rolling(4, min_periods=4).mean()
        total = out[["energy_2", "energy_4", "energy_8", "energy_16"]].sum(axis=1, min_count=4)
        out["coarse_share"] = out["energy_16"].div(total.where(total > 0))
        out["signed_energy"] = out["coarse_share"] * np.sign(out["haar_16"])
        out["calendar_slot"] = np.minimum((data.index.day - 1) // 6, 4)
        history = out["signed_energy"].shift(1)
        out["global_balance"] = history.rolling(504, min_periods=504).mean()
        out["seasonal_balance"] = np.nan
        for slot in range(5):
            member = out["calendar_slot"].eq(slot)
            valid = member.shift(1, fill_value=False) & history.notna()
            count = valid.astype(float).rolling(504, min_periods=504).sum()
            balance = history.where(valid, 0.0).rolling(504, min_periods=504).sum().div(count.where(count >= 20))
            out.loc[member, "seasonal_balance"] = balance.loc[member]
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["enter"] = (
            (out["haar_16"] > 0)
            & (out["coarse_share"] >= params.energy_share)
            & (out["seasonal_balance"] > 0.10)
            & (out["seasonal_balance"] > out["global_balance"] + 0.05)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        if not 0.0 < params.energy_share < 1.0 or params.trail_atr <= 0:
            raise ValueError("energy_share must be between zero and one; trail_atr must be positive")
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["enter"].fillna(False).to_numpy(dtype=bool)
        active = pending = armed = False
        entry = stop = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and i > 0 and np.isfinite(atrs[i - 1]):
                    entry = opens[i]
                    stop = entry - 3.0 * atrs[i - 1]
                    age = 0
                    armed = False
                    active = True
            if active:
                age += 1
                # A breach of the previous close's stop schedules a next-open exit.
                if (np.isfinite(lows[i]) and lows[i] <= stop) or age >= 18:
                    active = False
                    continue
                # Ratchet after testing the old stop to avoid intrabar ordering assumptions.
                if np.isfinite(closes[i]) and closes[i] >= entry * 1.02:
                    armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, closes[i] - params.trail_atr * atrs[i])
                signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = True
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    prices = np.full(24, 100.0)
    prices[2:4] = [103.0, 104.0]
    data = pd.DataFrame(
        {"open": 100.0, "high": prices + 1.0, "low": prices - 0.2,
         "close": prices, "volume": 1000.0}, index=index
    )
    indicators = pd.DataFrame({"atr": 1.0, "enter": False}, index=index)
    indicators.iloc[0, indicators.columns.get_loc("enter")] = True
    indicators.iloc[3:, indicators.columns.get_loc("atr")] = 4.0
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, SeasonalEnergyParams()).data
    assert result["signal"].iloc[:6].tolist() == [0, 1, 1, 1, 1, 0]
    data[["open", "high", "low", "close"]] = [100.0, 101.0, 99.8, 100.0]
    result = strategy.generate_signals(data, indicators, None, SeasonalEnergyParams()).data
    assert result["signal"].iloc[:20].tolist() == [0] + [1] * 18 + [0]
