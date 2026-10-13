from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    migration_threshold: float = 0.25
    breakeven_pct: float = 0.012
    trail_atr: float = 1.75


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791224364"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        price = np.log(data["close"].clip(lower=1e-9))
        out = pd.DataFrame(index=data.index)
        energies = {}
        for scale in (2, 4, 8):
            detail = price.rolling(scale).mean() - price.shift(scale).rolling(scale).mean()
            out[f"detail_{scale}"] = detail.fillna(0.0)
            energies[scale] = detail.pow(2).rolling(3).mean()
        total = sum(energies.values())
        centroid = sum(scale * energies[scale] for scale in energies) / total.replace(0.0, np.nan)
        out["migration"] = (centroid - centroid.shift(3)).fillna(0.0)
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = true_range.rolling(14).mean().fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        migration = indicators["migration"].to_numpy()
        detail_2 = indicators["detail_2"].to_numpy()
        detail_4 = indicators["detail_4"].to_numpy()
        detail_8 = indicators["detail_8"].to_numpy()
        atr = indicators["atr"].to_numpy()
        opens = data["open"].to_numpy()
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()
        raw = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        holding = False
        armed = False
        entry = stop = active_size = 0.0
        entry_bar = 0

        for i in range(len(data)):
            if i and raw[i - 1] == 1 and not holding:
                holding = True
                armed = False
                entry = float(opens[i])
                entry_bar = i
                active_size = sizes[i - 1]
                stop = max(0.01, entry - params.trail_atr * atr[i - 1])
            elif i and raw[i - 1] == 0 and holding:
                holding = False

            if holding:
                if lows[i] <= stop or i - entry_bar >= 3:
                    continue
                if highs[i] >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, highs[i] - params.trail_atr * atr[i])
                raw[i] = 1
                sizes[i] = active_size
            elif (
                i >= self.warmup_bars(params)
                and migration[i] >= params.migration_threshold
                and detail_4[i] > 0.0
                and detail_8[i] > abs(detail_2[i])
            ):
                active_size = 0.25 + 0.70 * min(migration[i] / 1.5, 1.0)
                raw[i] = 1
                sizes[i] = active_size

        df = pd.DataFrame({"signal": raw, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
