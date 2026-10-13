from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    surprise_z: float = 1.5
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791321337"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 132

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not np.isfinite(params.surprise_z) or params.surprise_z <= 0:
            raise ValueError("surprise_z must be positive and finite")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        log_close = out["log_close"]
        out["slope"] = (
            2.0 * log_close + log_close.shift(1)
            + 0.0 * log_close.shift(2)
            - log_close.shift(3) - 2.0 * log_close.shift(4)
        ) / 10.0
        out["innovation"] = out["slope"].diff()
        out["weekday"] = data.index.dayofweek
        out["seasonal_mean"] = np.nan
        out["seasonal_std"] = np.nan
        out["seasonal_count"] = 0.0
        valid = np.isfinite(out["innovation"])
        for weekday in range(5):
            selected = out["weekday"].eq(weekday)
            observed = selected & valid
            count = observed.astype(float).shift(1).rolling(126, min_periods=126).sum()
            samples = out["innovation"].where(observed, 0.0)
            total = samples.shift(1).rolling(126, min_periods=126).sum()
            squares = samples.pow(2).shift(1).rolling(126, min_periods=126).sum()
            mean = total / count.where(count >= 15)
            variance = (squares - total.pow(2) / count.where(count > 0)) / (count - 1).where(count >= 15)
            out.loc[selected, "seasonal_mean"] = mean.loc[selected]
            out.loc[selected, "seasonal_std"] = np.sqrt(variance.clip(lower=0)).loc[selected]
            out.loc[selected, "seasonal_count"] = count.loc[selected]
        out["surprise"] = (
            (out["innovation"] - out["seasonal_mean"])
            / out["seasonal_std"].where(out["seasonal_std"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        ready = (
            indicators["surprise"].notna()
            & np.isfinite(indicators["atr"])
            & indicators["atr"].gt(0)
            & indicators["weekday"].lt(5)
        )
        candidates = np.select([
            ready & indicators["surprise"].le(-params.surprise_z),
            ready & indicators["surprise"].ge(params.surprise_z),
        ], [1, -1], default=0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        side = 0
        pending_entry = False
        entry_price = 0.0
        stop_distance = 0.0
        held_sessions = 0
        for i in range(len(data)):
            if side != 0:
                if pending_entry:
                    entry_price = opens[i]
                    pending_entry = False
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        side = 0
                        continue
                held_sessions += 1
                stopped = (
                    not np.isfinite(closes[i])
                    or (side == 1 and closes[i] < entry_price - stop_distance)
                    or (side == -1 and closes[i] > entry_price + stop_distance)
                )
                if stopped or held_sessions >= 2:
                    side = 0
                    held_sessions = 0
                signals[i] = side
                continue
            if candidates[i] != 0:
                side = int(candidates[i])
                stop_distance = params.stop_atr * atr[i]
                pending_entry = True
                held_sessions = 0
                signals[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
