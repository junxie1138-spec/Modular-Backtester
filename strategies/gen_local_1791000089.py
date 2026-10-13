from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeShockwaveParams:
    history: int = 126
    energy_window: int = 16
    memory: int = 18
    compression_quantile: float = 0.40


class GeneratedStrategy(BaseStrategy[RangeShockwaveParams]):
    strategy_id = "gen_local_1791000089"

    @classmethod
    def params_type(cls) -> type[RangeShockwaveParams]:
        return RangeShockwaveParams

    @staticmethod
    def warmup_bars(params: RangeShockwaveParams) -> int:
        return int(2 * params.history + params.energy_window + params.memory + 8)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeShockwaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        high = data["high"].where(data["high"] > 0)
        low = data["low"].where(data["low"] > 0)
        out["log_range"] = np.log(high / low).where(high >= low)

        energy_columns = []
        for half in (1, 2, 4):
            name = f"haar_{2 * half}"
            recent = out["log_range"].rolling(half, min_periods=half).mean()
            out[name] = np.sqrt(half / 2.0) * (recent - recent.shift(half))
            energy_name = f"energy_{2 * half}"
            out[energy_name] = out[name].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
            energy_columns.append(energy_name)

        out["wavelet_energy"] = out[energy_columns].sum(axis=1, min_count=3)
        out["range_level"] = out["log_range"].rolling(8, min_periods=8).mean()
        out["energy_cutoff"] = out["wavelet_energy"].rolling(
            params.history, min_periods=params.history
        ).quantile(params.compression_quantile).shift(1)
        out["range_cutoff"] = out["range_level"].rolling(
            params.history, min_periods=params.history
        ).quantile(params.compression_quantile).shift(1)

        valid_compression = out[[
            "wavelet_energy", "range_level", "energy_cutoff", "range_cutoff"
        ]].notna().all(axis=1)
        compressed = (
            (out["wavelet_energy"] <= out["energy_cutoff"])
            & (out["range_level"] <= out["range_cutoff"])
        )
        out["compressed"] = compressed.astype(float).where(valid_compression)
        out["compression_occupancy"] = out["compressed"].rolling(
            params.memory, min_periods=params.memory
        ).mean()
        out["occupancy_cutoff"] = out["compression_occupancy"].rolling(
            params.history, min_periods=params.history
        ).quantile(0.50).shift(1)

        spread = data["high"] - data["low"]
        location = (
            (2.0 * data["close"] - data["high"] - data["low"])
            / spread.where(spread > 0)
        ).clip(-1.0, 1.0)
        out["close_location"] = location.mask(spread == 0, 0.0)
        out["valid_volume"] = data["volume"].where(data["volume"] >= 0)
        out["pressure_flow"] = out["close_location"] * out["valid_volume"]
        out["pressure_sum"] = out["pressure_flow"].rolling(
            params.memory, min_periods=params.memory
        ).sum()
        out["volume_sum"] = out["valid_volume"].rolling(
            params.memory, min_periods=params.memory
        ).sum()
        out["closing_pressure"] = out["pressure_sum"] / out["volume_sum"].where(
            out["volume_sum"] > 0
        )
        out["pressure_cutoff"] = out["closing_pressure"].rolling(
            params.history, min_periods=params.history
        ).quantile(0.50).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeShockwaveParams,
    ) -> SignalFrame:
        required = [
            "compression_occupancy", "occupancy_cutoff",
            "closing_pressure", "pressure_cutoff"
        ]
        valid = indicators[required].notna().all(axis=1)
        entry_condition = (
            valid
            & (indicators["compression_occupancy"] > indicators["occupancy_cutoff"])
            & (indicators["closing_pressure"] > indicators["pressure_cutoff"])
        )
        # The same predicate controls entry and continued holding;
        # its reversal is the sole exit rule.
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry_condition.fillna(False).astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
