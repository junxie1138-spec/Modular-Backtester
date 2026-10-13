from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    coarse_share_min: float = 0.55
    autocorrelation_min: float = 0.10
    profit_target: float = 0.006
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790989922"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["return"] = np.log(close).diff()
        r = out["return"]
        out["haar_fine"] = (r - r.shift(1)) / np.sqrt(2.0)
        out["haar_medium"] = (
            r + r.shift(1) - r.shift(2) - r.shift(3)
        ) / 2.0
        out["recent_sum"] = r.rolling(4, min_periods=4).sum()
        out["earlier_sum"] = r.shift(4).rolling(4, min_periods=4).sum()
        out["haar_coarse"] = (
            out["recent_sum"] - out["earlier_sum"]
        ) / np.sqrt(8.0)
        fine = out["haar_fine"]
        medium = out["haar_medium"]
        out["fine_energy"] = (
            fine.pow(2) + fine.shift(2).pow(2)
            + fine.shift(4).pow(2) + fine.shift(6).pow(2)
        )
        out["medium_energy"] = medium.pow(2) + medium.shift(4).pow(2)
        out["coarse_energy"] = out["haar_coarse"].pow(2)
        out["detail_energy"] = (
            out["fine_energy"] + out["medium_energy"] + out["coarse_energy"]
        )
        out["coarse_share"] = out["coarse_energy"] / out["detail_energy"].where(
            out["detail_energy"] > 1e-16
        )
        out["autocorrelation"] = r.rolling(6, min_periods=6).corr(r.shift(1))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        valid = indicators.notna().all(axis=1)
        entry = (
            valid
            & (indicators["coarse_share"] >= params.coarse_share_min)
            & (indicators["autocorrelation"] >= params.autocorrelation_min)
            & (indicators["haar_coarse"] > 0.0)
            & (indicators["recent_sum"] > 0.0)
            & (indicators["return"] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        hold_bars = max(1, min(2, int(params.hold_bars)))
        target = max(0.0001, float(params.profit_target))
        active = False
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if active:
                # A preceding close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                age = i - entry_bar + 1
                price_valid = np.isfinite(entry_price) and entry_price > 0.0
                target_hit = (
                    price_valid and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + target)
                )
                # Close-confirmed exits execute at the following open.
                if not price_valid or target_hit or age >= hold_bars:
                    active = False
                else:
                    signal[i] = 1
            elif entry[i]:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
