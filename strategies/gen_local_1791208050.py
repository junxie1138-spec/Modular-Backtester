from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoilParams:
    range_lookback: int = 20
    expansion_ratio: float = 1.8
    contraction_ratio: float = 0.65
    tail_fraction: float = 0.2
    minimum_recoil: float = 0.12
    maximum_recoil: float = 0.45
    profit_target: float = 0.012
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoilParams]):
    strategy_id = "gen_local_1791208050"

    @classmethod
    def params_type(cls) -> type[RecoilParams]:
        return RecoilParams

    def warmup_bars(self, params: RecoilParams) -> int:
        return max(200, max(1, int(params.range_lookback)) + 2)

    def indicators(self, data: pd.DataFrame, params: RecoilParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        window = max(1, int(params.range_lookback))
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["range_baseline"] = out["range"].rolling(window, min_periods=window).mean().shift(1)
        out["expansion"] = out["range"] / out["range_baseline"].where(out["range_baseline"] > 0)
        out["close_location"] = (data["close"] - data["low"]) / out["range"]
        out["prior_expansion"] = out["expansion"].shift(1)
        out["prior_close_location"] = out["close_location"].shift(1)
        out["prior_range"] = out["range"].shift(1)
        out["contraction"] = out["range"] / out["prior_range"]
        out["recoil"] = data["close"].diff() / out["prior_range"]
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoilParams,
    ) -> SignalFrame:
        ind = indicators
        valid = ind.notna().all(axis=1) & np.isfinite(data["close"]) & (data["close"] > 0)
        common = valid & (ind["prior_expansion"] >= params.expansion_ratio) & (ind["contraction"] <= params.contraction_ratio)
        long_entry = common & (ind["prior_close_location"] <= params.tail_fraction) & (data["close"] > ind["ma200"]) & ind["recoil"].between(params.minimum_recoil, params.maximum_recoil)
        short_entry = common & (ind["prior_close_location"] >= 1.0 - params.tail_fraction) & (data["close"] < ind["ma200"]) & (-ind["recoil"]).between(params.minimum_recoil, params.maximum_recoil)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        holding_bars = min(5, max(3, int(params.holding_bars)))
        profit_target = max(0.0001, float(params.profit_target))

        for i in range(len(data)):
            if position:
                if np.isnan(entry_price):
                    # The preceding close's decision fills at this bar's open.
                    entry_price = opens[i]
                    entry_bar = i
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                        entry_price = np.nan
                        continue
                gain = position * (closes[i] / entry_price - 1.0)
                elapsed = i - entry_bar + 1
                if (np.isfinite(gain) and gain >= profit_target) or elapsed >= holding_bars:
                    position = 0
                    entry_price = np.nan
                    continue
            elif entries[i]:
                position = int(entries[i])
                entry_price = np.nan
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
