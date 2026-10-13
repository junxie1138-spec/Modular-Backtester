from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAcceptanceParams:
    regime_window: int = 20
    pressure_window: int = 5
    gap_threshold: float = 0.8
    regime_threshold: float = 0.15
    pressure_threshold: float = 0.1
    profit_target: float = 0.012
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapAcceptanceParams]):
    strategy_id = "gen_local-w2_1790987741"

    @classmethod
    def params_type(cls) -> type[GapAcceptanceParams]:
        return GapAcceptanceParams

    def warmup_bars(self, params: GapAcceptanceParams) -> int:
        return int(max(max(2, params.regime_window) + 2,
                       max(1, params.pressure_window)))

    def indicators(self, data: pd.DataFrame, params: GapAcceptanceParams) -> pd.DataFrame:
        window = max(2, int(params.regime_window))
        pressure_window = max(1, int(params.pressure_window))
        prior_close = data["close"].shift(1).where(lambda s: s > 0)
        opening = data["open"].where(lambda s: s > 0)
        gap = data["open"].div(prior_close).sub(1)
        intraday = data["close"].div(opening).sub(1)
        gap_scale = gap.abs().rolling(window, min_periods=window).mean().shift(1)
        acceptance = gap.shift(1).rolling(window, min_periods=window).corr(intraday.shift(1))
        span = (data["high"] - data["low"]).where(lambda s: s > 0)
        close_location = ((2 * data["close"] - data["high"] - data["low"]) / span).clip(-1, 1).fillna(0)
        volume = data["volume"].clip(lower=0).fillna(0)
        signed_volume = close_location * volume
        volume_sum = volume.rolling(pressure_window, min_periods=pressure_window).sum()
        pressure = signed_volume.rolling(pressure_window, min_periods=pressure_window).sum().div(volume_sum.where(volume_sum > 0))
        normalized_gap = gap.div(gap_scale.where(gap_scale > 0))
        ready = (np.isfinite(normalized_gap) & np.isfinite(acceptance)
                 & np.isfinite(pressure) & np.isfinite(intraday))
        result = pd.DataFrame(index=data.index)
        result["gap"] = gap
        result["intraday"] = intraday
        result["gap_scale"] = gap_scale
        result["acceptance"] = acceptance
        result["close_location"] = close_location
        result["volume"] = volume
        result["signed_volume"] = signed_volume
        result["volume_sum"] = volume_sum
        result["pressure"] = pressure
        result["normalized_gap"] = normalized_gap
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["ready"] = ready.fillna(False)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapAcceptanceParams) -> SignalFrame:
        threshold = max(0.0, float(params.gap_threshold))
        regime_threshold = max(0.0, float(params.regime_threshold))
        continuation = ((indicators["acceptance"] > regime_threshold)
                        & (indicators["normalized_gap"] > threshold))
        recovery = ((indicators["acceptance"] < -regime_threshold)
                    & (indicators["normalized_gap"] < -threshold))
        gap_agrees = continuation | recovery
        pressure_agrees = indicators["pressure"] > max(0.0, float(params.pressure_threshold))
        entries = (indicators["ready"] & gap_agrees & pressure_agrees).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        pending_entry = False
        active = False
        entry_price = 0.0
        age = 0
        holding_limit = min(2, max(1, int(params.hold_bars)))
        target = max(0.0, float(params.profit_target))

        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i]
                    age = 0
                    active = True
                else:
                    continue

            if active:
                age += 1
                reached_target = (np.isfinite(closes[i])
                                  and closes[i] >= entry_price * (1.0 + target))
                # A closing exit decision executes at the following open.
                if reached_target or age >= holding_limit:
                    active = False
                    decisions[i] = 0
                else:
                    decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1
                pending_entry = True

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
