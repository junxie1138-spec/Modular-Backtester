from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TideGaugeParams:
    band_len: int = 20
    band_k: float = 2.0
    ma_len: int = 200
    entry_floor: float = 0.15
    entry_ceiling: float = 0.5
    profit_target: float = 0.03
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[TideGaugeParams]):
    """Long-only volatility-band relative-position reclaim.

    Treats price as a tide rising and falling inside a standard-deviation
    envelope. %B measures the relative position of close between the lower
    and upper band. An entry fires when %B crosses UP out of the low-tide
    zone while the 200-day MA regime filter confirms an uptrend. Each long
    is closed at a fixed profit target or a time-stop, whichever comes first.
    """

    strategy_id = "gen_a1_1778911400"

    @classmethod
    def params_type(cls) -> type[TideGaugeParams]:
        return TideGaugeParams

    @staticmethod
    def warmup_bars(params: TideGaugeParams) -> int:
        return int(max(params.ma_len, params.band_len)) + 2

    def indicators(self, data: pd.DataFrame, params: TideGaugeParams) -> pd.DataFrame:
        close = data["close"].astype(float)

        mid = close.rolling(params.band_len).mean()
        vol = close.rolling(params.band_len).std(ddof=0)
        upper = mid + params.band_k * vol
        lower = mid - params.band_k * vol

        width = (upper - lower)
        width = width.where(width > 0.0)
        pctb = (close - lower) / width

        sma200 = close.rolling(params.ma_len).mean()

        out = pd.DataFrame(index=data.index)
        out["mid"] = mid
        out["vol"] = vol
        out["upper"] = upper
        out["lower"] = lower
        out["pctb"] = pctb
        out["sma200"] = sma200
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TideGaugeParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        pctb = indicators["pctb"].to_numpy(dtype=float)
        sma200 = indicators["sma200"].to_numpy(dtype=float)
        n = len(close)

        pctb_prev = np.full(n, np.nan, dtype=float)
        if n > 1:
            pctb_prev[1:] = pctb[:-1]

        floor = float(params.entry_floor)
        ceiling = float(params.entry_ceiling)
        pt = float(params.profit_target)
        max_hold = int(params.max_hold)

        signal = np.zeros(n, dtype=int)
        in_pos = False
        entry_price = 0.0
        entry_idx = 0

        for i in range(n):
            regime_ok = (not np.isnan(sma200[i])) and (close[i] > sma200[i])

            if not in_pos:
                valid = (not np.isnan(pctb[i])) and (not np.isnan(pctb_prev[i]))
                cross_up = (
                    valid
                    and pctb_prev[i] <= floor
                    and pctb[i] > floor
                    and pctb[i] < ceiling
                )
                if cross_up and regime_ok:
                    in_pos = True
                    entry_price = close[i]
                    entry_idx = i
                    signal[i] = 1
            else:
                bars_held = i - entry_idx
                hit_pt = close[i] >= entry_price * (1.0 + pt)
                hit_time = bars_held >= max_hold
                if hit_pt or hit_time:
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
