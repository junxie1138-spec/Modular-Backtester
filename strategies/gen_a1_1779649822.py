from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CapitulationSpringParams:
    lookback_bars: int = 60
    return_percentile: float = 0.10
    volume_percentile: float = 0.90
    profit_target: float = 0.03
    max_hold_bars: int = 5


class GeneratedStrategy(BaseStrategy[CapitulationSpringParams]):
    strategy_id = "gen_a1_1779649822"

    @classmethod
    def params_type(cls):
        return CapitulationSpringParams

    def warmup_bars(self, params: CapitulationSpringParams) -> int:
        return int(params.lookback_bars) + 1

    def indicators(self, data: pd.DataFrame, params: CapitulationSpringParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)
        lookback = int(params.lookback_bars)

        returns = close.pct_change()
        ret_rank = returns.rolling(lookback, min_periods=lookback).rank(pct=True)
        vol_rank = volume.rolling(lookback, min_periods=lookback).rank(pct=True)

        ind = pd.DataFrame(index=data.index)
        ind["returns"] = returns
        ind["ret_rank"] = ret_rank
        ind["vol_rank"] = vol_rank
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CapitulationSpringParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        ret_rank = indicators["ret_rank"].to_numpy(dtype=float)
        vol_rank = indicators["vol_rank"].to_numpy(dtype=float)

        ret_thr = float(params.return_percentile)
        vol_thr = float(params.volume_percentile)
        pt = float(params.profit_target)
        max_hold = int(params.max_hold_bars)

        raw_signal = np.zeros(n, dtype=np.int64)
        in_position = False
        entry_idx = -1
        entry_price = 0.0

        for i in range(n):
            if not in_position:
                rr = ret_rank[i]
                vr = vol_rank[i]
                if (
                    not np.isnan(rr)
                    and not np.isnan(vr)
                    and rr <= ret_thr
                    and vr >= vol_thr
                ):
                    in_position = True
                    entry_idx = i
                    entry_price = close[i]
                    raw_signal[i] = 1
            else:
                bars_held = i - entry_idx
                if entry_price > 0.0:
                    pnl_pct = (close[i] / entry_price) - 1.0
                else:
                    pnl_pct = 0.0
                if pnl_pct >= pt or bars_held >= max_hold:
                    raw_signal[i] = 0
                    in_position = False
                    entry_idx = -1
                    entry_price = 0.0
                else:
                    raw_signal[i] = 1

        signal = pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        size = pd.Series(1.0, index=data.index, dtype=float)

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
