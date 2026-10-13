from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTransmissionParams:
    lookback: int = 6
    gap_multiple: float = 1.25
    coupling_threshold: float = 0.25
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapTransmissionParams]):
    strategy_id = "gen_local_1790996397"

    @classmethod
    def params_type(cls) -> type[GapTransmissionParams]:
        return GapTransmissionParams

    def warmup_bars(self, params: GapTransmissionParams) -> int:
        return min(8, max(2, int(params.lookback))) + 2

    def indicators(self, data: pd.DataFrame, params: GapTransmissionParams) -> pd.DataFrame:
        window = min(8, max(2, int(params.lookback)))
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["gap"] = data["open"] / out["prior_close"].where(out["prior_close"] > 0) - 1.0
        out["daytime_return"] = data["close"] / data["open"].where(data["open"] > 0) - 1.0
        out["gap_daytime_product"] = out["gap"] * out["daytime_return"]
        out["gap_squared"] = out["gap"] ** 2
        out["historical_gap_scale"] = out["gap"].abs().shift(1).rolling(window, min_periods=window).mean()
        out["historical_transmission"] = out["gap_daytime_product"].shift(1).rolling(window, min_periods=window).mean()
        out["historical_gap_energy"] = out["gap_squared"].shift(1).rolling(window, min_periods=window).mean()
        out["coupling"] = out["historical_transmission"] / out["historical_gap_energy"].where(out["historical_gap_energy"] > 0)
        out["relative_gap"] = out["gap"].abs() / out["historical_gap_scale"].where(out["historical_gap_scale"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapTransmissionParams) -> SignalFrame:
        valid = indicators[["gap", "coupling", "relative_gap"]].notna().all(axis=1)
        eligible = valid & (indicators["relative_gap"] >= max(0.0, float(params.gap_multiple)))
        eligible &= indicators["coupling"].abs() >= max(0.0, float(params.coupling_threshold))
        directions = np.sign(indicators["gap"]) * np.sign(indicators["coupling"])
        candidates = directions.where(eligible, 0).fillna(0).astype(int).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        profit_target = max(0.0001, float(params.profit_target))
        hold_limit = min(2, max(1, int(params.max_hold_bars)))
        position = 0
        entry_price = np.nan
        held_bars = 0

        for i in range(len(data)):
            if position != 0:
                # A previous close's entry decision fills at this bar's open.
                if np.isnan(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        held_bars = 0
                        continue
                    entry_price = opens[i]
                held_bars += 1
                gain = position * (closes[i] / entry_price - 1.0)
                # Close-observed exits also fill at the following open.
                if (np.isfinite(gain) and gain >= profit_target) or held_bars >= hold_limit:
                    position = 0
                    entry_price = np.nan
                    held_bars = 0
                    continue
            elif candidates[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                entry_price = np.nan
                held_bars = 0
            decisions[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
