from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA2Params:
    rank_window: int = 50
    entry_pct: float = 0.95
    hold_bars: int = 4
    refractory_bars: int = 5
    trend_ma: int = 100
    use_trend_gate: bool = True


class GeneratedStrategy(BaseStrategy[GenA2Params]):
    strategy_id = "gen_a2_1779148675"

    @classmethod
    def params_type(cls) -> type[GenA2Params]:
        return GenA2Params

    @staticmethod
    def warmup_bars(params: GenA2Params) -> int:
        return int(max(params.rank_window, params.trend_ma)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GenA2Params) -> pd.DataFrame:
        close = data["close"]
        ind = pd.DataFrame(index=data.index)
        win = max(2, int(params.rank_window))
        # Rolling percentile rank of close within the trailing window (0..1).
        ind["rank"] = close.rolling(win).rank(pct=True)
        ma_win = max(2, int(params.trend_ma))
        ind["trend_ma"] = close.rolling(ma_win).mean()
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA2Params,
    ) -> SignalFrame:
        close = data["close"]
        rank = indicators["rank"]
        ma = indicators["trend_ma"]

        thr = float(params.entry_pct)
        breakout = rank >= thr
        if params.use_trend_gate:
            breakout = breakout & (close > ma)
        breakout = breakout.fillna(False)

        # Two-bar confirmation: the close must hold in the top percentile band
        # on this bar AND the immediately prior bar before an entry is armed.
        confirmed = (breakout & breakout.shift(1).fillna(False)).to_numpy()

        n = len(data)
        raw = np.zeros(n, dtype=int)
        hold_bars = max(1, int(params.hold_bars))
        refr = max(0, int(params.refractory_bars))

        in_pos = False
        held = 0
        cooldown = 0
        for i in range(n):
            if in_pos:
                raw[i] = 1
                held += 1
                if held >= hold_bars:
                    in_pos = False
                    cooldown = refr
                continue
            if cooldown > 0:
                cooldown -= 1
                continue
            if confirmed[i]:
                in_pos = True
                held = 1
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
