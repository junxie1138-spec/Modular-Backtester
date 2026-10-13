from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MidmonthAbsorptionParams:
    volume_lookback: int = 20
    volume_multiple: float = 1.15
    calendar_start: int = 14
    calendar_end: int = 18
    close_location_min: float = 0.75
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[MidmonthAbsorptionParams]):
    strategy_id = "gen_local_1790990070"

    @classmethod
    def params_type(cls) -> type[MidmonthAbsorptionParams]:
        return MidmonthAbsorptionParams

    def warmup_bars(self, params: MidmonthAbsorptionParams) -> int:
        return int(max(200, params.volume_lookback + 1))

    def indicators(self, data: pd.DataFrame, params: MidmonthAbsorptionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["ma_200"] = data["close"].rolling(200, min_periods=200).mean()
        out["previous_close"] = data["close"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["volume_baseline"] = data["volume"].shift(1).rolling(
            params.volume_lookback, min_periods=params.volume_lookback
        ).mean()
        out["relative_volume"] = data["volume"].div(
            out["volume_baseline"].where(out["volume_baseline"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["intraday_return"] = data["close"].div(
            data["open"].where(data["open"] > 0)
        ).sub(1).replace([np.inf, -np.inf], np.nan)
        out["close_location"] = (data["close"] - data["low"]).div(
            (data["high"] - data["low"]).where(data["high"] > data["low"])
        ).replace([np.inf, -np.inf], np.nan)
        out["calendar_window"] = (
            (data.index.day >= params.calendar_start)
            & (data.index.day <= params.calendar_end)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MidmonthAbsorptionParams,
    ) -> SignalFrame:
        ready = indicators[
            ["ma_200", "previous_close", "previous_high", "relative_volume",
             "intraday_return", "close_location"]
        ].notna().all(axis=1)
        entries = (
            ready
            & indicators["calendar_window"]
            & (data["close"] > indicators["ma_200"])
            & (data["open"] < indicators["previous_close"])
            & (data["close"] > indicators["previous_high"])
            & (indicators["intraday_return"] > 0)
            & (indicators["relative_volume"] >= params.volume_multiple)
            & (indicators["close_location"] >= params.close_location_min)
        ).fillna(False).to_numpy(dtype=bool)

        # Entry decisions are shifted below; expiry shifts identically.
        # Ignore new entry conditions while holding, including the expiry bar.
        signal = np.zeros(len(data), dtype=np.int64)
        remaining = 0
        holding_bars = max(1, int(params.holding_bars))
        for i in range(len(data)):
            if remaining > 0:
                remaining -= 1
                if remaining > 0:
                    signal[i] = 1
            elif entries[i]:
                remaining = holding_bars
                signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
