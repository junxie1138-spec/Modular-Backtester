from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    streak_k: int = 3
    vol_window: int = 20
    target_vol: float = 0.15
    max_size: float = 1.0
    min_size: float = 0.2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778885270"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.vol_window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        ret = data["close"].pct_change()

        up = (ret > 0).astype(int)
        down = (ret < 0).astype(int)

        # consecutive up-close streak length ending at each bar
        up_grp = (up != up.shift()).cumsum()
        up_streak = up.groupby(up_grp).cumcount().add(1).where(up == 1, 0)

        # consecutive down-close streak length ending at each bar (symmetric mirror)
        down_grp = (down != down.shift()).cumsum()
        down_streak = down.groupby(down_grp).cumcount().add(1).where(down == 1, 0)

        out["up_streak"] = up_streak.fillna(0).astype(float)
        out["down_streak"] = down_streak.fillna(0).astype(float)

        # annualized realized volatility -> inverse-vol target size
        realized_vol = ret.rolling(int(params.vol_window)).std() * np.sqrt(252.0)
        vol_size = float(params.target_vol) / realized_vol.replace(0.0, np.nan)
        vol_size = vol_size.clip(lower=float(params.min_size), upper=float(params.max_size))
        out["vol_size"] = vol_size.fillna(float(params.min_size))

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        k = int(params.streak_k)
        up_streak = indicators["up_streak"].to_numpy()
        down_streak = indicators["down_streak"].to_numpy()

        n = len(df)
        raw = np.zeros(n, dtype=np.int64)
        pos = 0
        for i in range(n):
            if pos == 0:
                # entry: K consecutive up-closes
                if up_streak[i] >= k:
                    pos = 1
            else:
                # symmetric signal-reversal exit: K consecutive down-closes
                if down_streak[i] >= k:
                    pos = 0
            raw[i] = pos

        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size = indicators["vol_size"].astype(float)
        size = size.clip(lower=float(params.min_size), upper=float(params.max_size))
        df["size"] = size.fillna(float(params.min_size))

        return SignalFrame(data=df, signal_column="signal", size_column="size")
