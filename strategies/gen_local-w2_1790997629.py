from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletPositionParams:
    position_window: int = 32
    energy_window: int = 16
    coarse_share: float = 0.65
    tail_fraction: float = 0.20
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveletPositionParams]):
    strategy_id = "gen_local-w2_1790997629"

    @classmethod
    def params_type(cls) -> type[WaveletPositionParams]:
        return WaveletPositionParams

    def warmup_bars(self, params: WaveletPositionParams) -> int:
        return int(max(params.position_window, params.energy_window + 7))

    def indicators(self, data: pd.DataFrame, params: WaveletPositionParams) -> pd.DataFrame:
        if params.position_window < 2 or params.energy_window < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not 0.0 < params.coarse_share < 1.0:
            raise ValueError("coarse_share must be between zero and one")
        if not 0.0 < params.tail_fraction < 0.5:
            raise ValueError("tail_fraction must be between zero and one-half")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for half in (1, 2, 4):
            out[f"recent_{half}"] = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"prior_{half}"] = out[f"recent_{half}"].shift(half)
            out[f"haar_{half}"] = np.sqrt(half / 2.0) * (
                out[f"recent_{half}"] - out[f"prior_{half}"]
            )
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out["energy_1"] + out["energy_2"] + out["energy_4"]
        out["coarse_fraction"] = out["energy_4"] / out["total_energy"].where(
            out["total_energy"] > 0
        )
        out["channel_low"] = data["low"].rolling(
            params.position_window, min_periods=params.position_window
        ).min()
        out["channel_high"] = data["high"].rolling(
            params.position_window, min_periods=params.position_window
        ).max()
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["relative_position"] = (
            data["close"] - out["channel_low"]
        ) / out["channel_width"].where(out["channel_width"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletPositionParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two")
        ready = indicators.notna().all(axis=1)
        coherent = ready & indicators["coarse_fraction"].ge(params.coarse_share)
        long_entry = (
            coherent
            & indicators["relative_position"].ge(1.0 - params.tail_fraction)
            & indicators["haar_4"].gt(0.0)
        )
        short_entry = (
            coherent
            & indicators["relative_position"].le(params.tail_fraction)
            & indicators["haar_4"].lt(0.0)
        )
        candidate = pd.Series(
            np.select([long_entry, short_entry], [1, -1], default=0),
            index=data.index,
            dtype=int,
        )
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        for i, direction in enumerate(confirmed.to_numpy()):
            if side != 0:
                if i - entry_bar >= params.hold_bars:
                    side = 0
            elif direction != 0:
                side = int(direction)
                entry_bar = i
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
