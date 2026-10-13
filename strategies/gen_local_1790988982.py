from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureMomentumParams:
    area_window: int = 12
    area_threshold: float = 0.06
    energy_window: int = 5
    max_downside_share: float = 0.25
    profit_target: float = 0.02
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[SignatureMomentumParams]):
    strategy_id = "gen_local_1790988982"

    @classmethod
    def params_type(cls) -> type[SignatureMomentumParams]:
        return SignatureMomentumParams

    def warmup_bars(self, params: SignatureMomentumParams) -> int:
        return int(max(2, params.area_window, params.energy_window) + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureMomentumParams) -> pd.DataFrame:
        length = max(2, int(params.area_window))
        energy_length = max(2, int(params.energy_window))
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(np.isfinite(close) & (close > 0)))
        out["log_return"] = out["log_close"].diff()
        out["displacement"] = out["log_close"] - out["log_close"].shift(length)
        # Antisymmetric level-two signature for unit time increments.
        # Equivalent to half the sum of (2*j-length-1)*return_j.
        out["signature_area"] = (
            0.5 * (length - 1)
            * (out["log_close"] + out["log_close"].shift(length))
            - out["log_close"].shift(1).rolling(
                length - 1, min_periods=length - 1
            ).sum()
        )
        out["path_length"] = out["log_return"].abs().rolling(
            length, min_periods=length
        ).sum()
        out["signature_score"] = out["signature_area"] / (
            length * out["path_length"].where(out["path_length"] > 0)
        )
        out["return_energy"] = out["log_return"].pow(2)
        out["downside_energy"] = out["log_return"].clip(upper=0).pow(2)
        out["total_energy"] = out["return_energy"].rolling(
            energy_length, min_periods=energy_length
        ).sum()
        out["total_downside_energy"] = out["downside_energy"].rolling(
            energy_length, min_periods=energy_length
        ).sum()
        out["downside_share"] = out["total_downside_energy"] / out[
            "total_energy"
        ].where(out["total_energy"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureMomentumParams,
    ) -> SignalFrame:
        # Primitive one: upward path curvature with positive net movement.
        curvature_agrees = (
            (indicators["signature_score"] > params.area_threshold)
            & (indicators["displacement"] > 0)
        )
        # Primitive two: recent squared returns are dominated by advances.
        energy_agrees = (
            indicators["downside_share"] < params.max_downside_share
        )
        entries = (curvature_agrees & energy_agrees).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        entry_price = np.nan
        held_bars = 0
        holding_limit = max(1, int(params.max_hold_bars))
        target = max(0.0, float(params.profit_target))

        for i in range(len(data)):
            if pending_entry:
                # Yesterday's decision fills at today's open; never read a future bar.
                pending_entry = False
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    active = False
                    continue
                entry_price = opens[i]
                held_bars = 0

            if active:
                held_bars += 1
                reached_target = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + target)
                )
                if reached_target or held_bars >= holding_limit:
                    # This closing decision exits at the following bar's fill.
                    active = False
                    raw[i] = 0
                else:
                    raw[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                pending_entry = True
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
