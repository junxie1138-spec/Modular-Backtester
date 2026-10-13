from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DamageCycleParams:
    peak_window: int = 20
    damage_window: int = 10
    percentile_window: int = 126
    depth_percentile: float = 0.80
    regime_percentile: float = 0.65
    profit_target: float = 0.006
    max_holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[DamageCycleParams]):
    strategy_id = "gen_local-w2_1791000612"

    @classmethod
    def params_type(cls) -> type[DamageCycleParams]:
        return DamageCycleParams

    def warmup_bars(self, params: DamageCycleParams) -> int:
        return int(max(params.peak_window, params.damage_window + 1) + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: DamageCycleParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = (1.0 - close / out["peak"]).clip(lower=0.0)
        out["depth_threshold"] = out["depth"].rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.depth_percentile).shift(1)
        out["overnight_return"] = opening / close.shift(1) - 1.0
        out["intraday_return"] = close / opening - 1.0
        out["overnight_loss"] = (-out["overnight_return"]).clip(lower=0.0)
        out["intraday_loss"] = (-out["intraday_return"]).clip(lower=0.0)
        out["overnight_damage"] = out["overnight_loss"].rolling(params.damage_window, min_periods=params.damage_window).sum()
        out["intraday_damage"] = out["intraday_loss"].rolling(params.damage_window, min_periods=params.damage_window).sum()
        out["total_damage"] = out["overnight_damage"] + out["intraday_damage"]
        out["intraday_share"] = out["intraday_damage"] / out["total_damage"].where(out["total_damage"] > 0)
        share_history = out["intraday_share"].rolling(params.percentile_window, min_periods=params.percentile_window)
        out["regime_lower"] = share_history.quantile(1.0 - params.regime_percentile).shift(1)
        out["regime_upper"] = share_history.quantile(params.regime_percentile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DamageCycleParams) -> SignalFrame:
        required = ["depth", "depth_threshold", "intraday_share", "regime_lower", "regime_upper"]
        valid = indicators[required].notna().all(axis=1)
        extreme = valid & (indicators["depth"] > indicators["depth_threshold"])
        onset = extreme & ~extreme.shift(1, fill_value=False)
        long_entry = onset & (indicators["intraday_share"] < indicators["regime_lower"])
        short_entry = onset & (indicators["intraday_share"] > indicators["regime_upper"])
        candidates = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        direction = 0
        entry_price = np.nan
        held_bars = 0
        holding_limit = max(1, min(2, int(params.max_holding_bars)))

        for i in range(len(data)):
            if direction != 0:
                # The preceding close's entry decision fills at this bar's open.
                if held_bars == 0:
                    entry_price = opening[i]
                held_bars += 1
                if not np.isfinite(entry_price) or entry_price <= 0:
                    direction = 0
                    entry_price = np.nan
                    held_bars = 0
                    continue
                gain = direction * (closing[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                if target_hit or held_bars >= holding_limit:
                    direction = 0
                    entry_price = np.nan
                    held_bars = 0
                decisions[i] = direction
            elif candidates[i] != 0 and np.isfinite(closing[i]) and closing[i] > 0:
                direction = int(candidates[i])
                held_bars = 0
                decisions[i] = direction

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
