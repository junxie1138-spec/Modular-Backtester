from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletBreakoutParams:
    coarse_share: float = 0.60
    energy_growth: float = 1.10
    profit_target: float = 0.012


class GeneratedStrategy(BaseStrategy[WaveletBreakoutParams]):
    strategy_id = "gen_local_1791330517"

    @classmethod
    def params_type(cls) -> type[WaveletBreakoutParams]:
        return WaveletBreakoutParams

    def warmup_bars(self, params: WaveletBreakoutParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: WaveletBreakoutParams) -> pd.DataFrame:
        if not 0.0 < params.coarse_share < 1.0:
            raise ValueError("coarse_share must be between zero and one")
        if not np.isfinite(params.energy_growth) or params.energy_growth <= 0.0:
            raise ValueError("energy_growth must be positive and finite")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be positive and finite")
        out = pd.DataFrame(index=data.index)
        log_price = np.log(data["close"].where(data["close"] > 0.0))
        x = [log_price.shift(k) for k in range(8)]
        out["log_price"] = log_price
        for j in range(4):
            out[f"fine_{j}"] = (x[2 * j] - x[2 * j + 1]) / np.sqrt(2.0)
        for j in range(2):
            k = 4 * j
            out[f"medium_{j}"] = (x[k] + x[k + 1] - x[k + 2] - x[k + 3]) / 2.0
        out["coarse"] = (x[0] + x[1] + x[2] + x[3] - x[4] - x[5] - x[6] - x[7]) / np.sqrt(8.0)
        out["fine_energy"] = sum(out[f"fine_{j}"] ** 2 for j in range(4))
        out["medium_energy"] = sum(out[f"medium_{j}"] ** 2 for j in range(2))
        out["coarse_energy"] = out["coarse"] ** 2
        out["total_energy"] = out["fine_energy"] + out["medium_energy"] + out["coarse_energy"]
        out["coarse_share"] = out["coarse_energy"] / out["total_energy"].where(out["total_energy"] > 0.0)
        out["previous_coarse_energy"] = out["coarse_energy"].shift(1)
        out["breakout_level"] = data["high"].shift(1).rolling(3, min_periods=3).max()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletBreakoutParams) -> SignalFrame:
        entries = (
            (data["close"] > indicators["breakout_level"])
            & (indicators["coarse"] > 0.0)
            & (indicators["coarse_share"] >= params.coarse_share)
            & (indicators["previous_coarse_energy"] > 0.0)
            & (indicators["coarse_energy"] >= params.energy_growth * indicators["previous_coarse_energy"])
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                target_hit = valid_entry and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_stop = i - entry_bar + 1 >= 2
                if not valid_entry or target_hit or time_stop:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1
                active = True
                entry_bar = i + 1
                entry_price = np.nan
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
