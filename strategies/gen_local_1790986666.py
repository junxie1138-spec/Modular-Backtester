from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeMarchParams:
    lookback: int = 12
    stop_atr: float = 2.6


class GeneratedStrategy(BaseStrategy[RangeMarchParams]):
    strategy_id = "gen_local_1790986666"

    @classmethod
    def params_type(cls) -> type[RangeMarchParams]:
        return RangeMarchParams

    def warmup_bars(self, params: RangeMarchParams) -> int:
        return max(15, max(3, int(params.lookback)) + 1)

    def indicators(self, data: pd.DataFrame, params: RangeMarchParams) -> pd.DataFrame:
        n = max(3, int(params.lookback))
        out = pd.DataFrame(index=data.index)
        out["width"] = (data["high"] - data["low"]).clip(lower=0)
        out["high_step"] = data["high"].diff()
        out["low_step"] = data["low"].diff()
        hi, lo = out["high_step"], out["low_step"]
        valid = hi.notna() & lo.notna()
        out["shared_step"] = pd.Series(
            np.where(
                (hi > 0) & (lo > 0), np.minimum(hi, lo),
                np.where((hi < 0) & (lo < 0), np.maximum(hi, lo), 0.0),
            ),
            index=data.index,
        ).where(valid)
        out["edge_travel"] = np.maximum(hi.abs(), lo.abs())
        out["shared_sum"] = out["shared_step"].rolling(n, min_periods=n).sum()
        out["travel_sum"] = out["edge_travel"].rolling(n, min_periods=n).sum()
        out["strength"] = out["shared_sum"] / out["travel_sum"].replace(0, np.nan)
        out["width_fast"] = out["width"].rolling(3, min_periods=3).mean()
        out["width_base"] = out["width"].rolling(n, min_periods=n).mean()
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [
                out["width"],
                (data["high"] - out["previous_close"]).abs(),
                (data["low"] - out["previous_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeMarchParams,
    ) -> SignalFrame:
        orderly = (
            (indicators["width_base"] > 0)
            & (indicators["width_fast"] <= 1.15 * indicators["width_base"])
        )
        entry = pd.Series(0, index=data.index, dtype=int)
        entry.loc[
            orderly & (indicators["strength"] > 0.55)
            & (indicators["shared_step"] > 0)
        ] = 1
        entry.loc[
            orderly & (indicators["strength"] < -0.55)
            & (indicators["shared_step"] < 0)
        ] = -1

        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = entry.to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        entry_atr = np.nan
        k = max(0.1, float(params.stop_atr))

        for i in range(len(data)):
            if position:
                # The preceding close decided entry; this bar's open is its fill reference.
                if i == entry_bar:
                    entry_price = opens[i]
                invalid = (
                    not np.isfinite(closes[i])
                    or not np.isfinite(entry_price)
                    or entry_price <= 0
                )
                stopped = (
                    not invalid
                    and position * (closes[i] - entry_price) < -k * entry_atr
                )
                expired = i - entry_bar + 1 >= 18
                if invalid or stopped or expired:
                    position = 0
            elif (
                i >= self.warmup_bars(params) - 1
                and candidates[i] != 0
                and np.isfinite(closes[i]) and closes[i] > 0
                and np.isfinite(atr[i]) and atr[i] > 0
            ):
                position = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
                entry_atr = atr[i]
            decisions[i] = position

        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
