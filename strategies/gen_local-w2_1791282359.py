from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    ma_window: int = 60
    energy_window: int = 6
    recovery_z: float = 1.5
    energy_ratio: float = 0.65
    profit_target: float = 0.04
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791282359"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.ma_window + 1, 2 * params.energy_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if params.ma_window < 2 or params.energy_window < 1:
            raise ValueError("Indicator windows must be positive; ma_window must be at least two.")
        if params.recovery_z <= 0 or not 0 < params.energy_ratio < 1:
            raise ValueError("Invalid recovery or energy threshold.")
        if params.profit_target <= 0 or params.max_hold_bars < 1:
            raise ValueError("Exit parameters must be positive.")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].replace(0.0, np.nan)
        out["previous_z"] = out["z"].shift(1)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["down_squared"] = out["return"].clip(upper=0.0).pow(2)
        out["up_squared"] = out["return"].clip(lower=0.0).pow(2)
        out["down_energy"] = out["down_squared"].rolling(params.energy_window).sum()
        out["up_energy"] = out["up_squared"].rolling(params.energy_window).sum()
        out["previous_down_energy"] = out["down_energy"].shift(params.energy_window)
        out["previous_up_energy"] = out["up_energy"].shift(params.energy_window)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        z = indicators["z"]
        previous_z = indicators["previous_z"]
        long_recovery = (previous_z <= -params.recovery_z) & (z > -params.recovery_z) & (z < 0.0)
        short_recovery = (previous_z >= params.recovery_z) & (z < params.recovery_z) & (z > 0.0)
        down_contracts = (indicators["previous_down_energy"] > 0.0) & (
            indicators["down_energy"] <= params.energy_ratio * indicators["previous_down_energy"]
        )
        up_contracts = (indicators["previous_up_energy"] > 0.0) & (
            indicators["up_energy"] <= params.energy_ratio * indicators["previous_up_energy"]
        )
        ready = indicators[["z", "previous_z", "down_energy", "up_energy",
                            "previous_down_energy", "previous_up_energy"]].notna().all(axis=1)
        entries = np.select(
            [(ready & long_recovery & down_contracts).to_numpy(),
             (ready & short_recovery & up_contracts).to_numpy()],
            [1, -1], default=0
        )
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        held_bars = 0
        for i in range(len(data)):
            if position != 0:
                # A decision from the preceding close fills at this bar's open.
                if np.isnan(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        position = 0
                        signals[i] = 0
                        continue
                    entry_price = opens[i]
                held_bars += 1
                gain = position * (closes[i] / entry_price - 1.0)
                if (np.isfinite(gain) and gain >= params.profit_target) or held_bars >= params.max_hold_bars:
                    position = 0
                    entry_price = np.nan
                    held_bars = 0
            elif entries[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(entries[i])
                entry_price = np.nan
                held_bars = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
