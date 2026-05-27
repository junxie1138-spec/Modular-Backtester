from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    dd_lookback: int = 20
    dd_min: float = 0.02
    dd_max: float = 0.08
    vol_lookback: int = 20
    snr_threshold: float = 1.0
    tom_days_before: int = 3
    tom_days_after: int = 3
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779654372"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return int(max(params.dd_lookback, params.vol_lookback)) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]

        rolling_high = close.rolling(
            params.dd_lookback, min_periods=params.dd_lookback
        ).max()
        dd_depth = (rolling_high - close) / rolling_high
        out["dd_depth"] = dd_depth

        returns = close.pct_change()
        rolling_vol = returns.rolling(
            params.vol_lookback, min_periods=params.vol_lookback
        ).std()
        out["rolling_vol"] = rolling_vol
        out["snr"] = dd_depth / rolling_vol.replace(0.0, np.nan)

        idx = data.index
        ym = pd.Series(idx.year.astype(np.int64) * 100 + idx.month.astype(np.int64), index=idx)
        tdom = ym.groupby(ym).cumcount() + 1
        size_per_month = ym.groupby(ym).transform("size")
        tdom_from_end = size_per_month - tdom + 1
        out["tdom"] = tdom.astype(np.int64)
        out["tdom_from_end"] = tdom_from_end.astype(np.int64)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        dd_depth = indicators["dd_depth"]
        snr = indicators["snr"]
        tdom = indicators["tdom"]
        tdom_from_end = indicators["tdom_from_end"]

        depth_ok = (dd_depth >= params.dd_min) & (dd_depth <= params.dd_max)
        snr_ok = snr >= params.snr_threshold
        season_ok = (tdom <= int(params.tom_days_after)) | (
            tdom_from_end <= int(params.tom_days_before)
        )

        entry = (depth_ok & snr_ok & season_ok).fillna(False).to_numpy(dtype=bool)

        n = len(df)
        raw = np.zeros(n, dtype=np.int64)
        remaining = 0
        hold = int(params.hold_bars)
        for i in range(n):
            if remaining > 0:
                raw[i] = 1
                remaining -= 1
            elif entry[i]:
                raw[i] = 1
                remaining = hold - 1

        df["signal"] = (
            pd.Series(raw, index=df.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
