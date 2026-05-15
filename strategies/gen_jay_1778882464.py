from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_threshold: int = 4
    efficiency_window: int = 20
    efficiency_plastic: float = 0.45
    profit_target_pct: float = 3.0
    time_stop_bars: int = 20


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778882464"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.efficiency_window + params.streak_threshold + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        delta = close.diff()

        # Vectorized consecutive up/down streak counts
        up = delta > 0
        dn = delta < 0
        up_streak = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
        up_streak = up_streak.where(up, 0)
        dn_streak = dn.groupby((dn != dn.shift()).cumsum()).cumcount() + 1
        dn_streak = dn_streak.where(dn, 0)
        # Positive = up run length, negative = down run length
        streak = up_streak - dn_streak

        # Kaufman Efficiency Ratio: |net displacement| / total path length
        # High ER = plastic (trending), Low ER = elastic (mean-reverting)
        net_disp = close.diff(params.efficiency_window).abs()
        path_len = delta.abs().rolling(params.efficiency_window).sum()
        efficiency = (net_disp / path_len.replace(0, np.nan)).clip(0.0, 1.0)

        plastic = (efficiency >= params.efficiency_plastic).fillna(False)
        thr = params.streak_threshold

        # Regime-adaptive signal polarity:
        # Plastic + up streak   -> momentum long
        # Plastic + down streak -> momentum short
        # Elastic + down streak -> reversal long
        # Elastic + up streak   -> reversal short
        raw = pd.Series(
            np.select(
                [
                    plastic & (streak >= thr),
                    plastic & (streak <= -thr),
                    ~plastic & (streak <= -thr),
                    ~plastic & (streak >= thr),
                ],
                [1, -1, 1, -1],
                default=0,
            ).astype(int),
            index=data.index,
        )

        ind = pd.DataFrame(index=data.index)
        ind["streak"] = streak
        ind["efficiency"] = efficiency
        ind["raw_signal"] = raw
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        raw = indicators["raw_signal"]

        # Hard twist: two-bar confirmation before any entry
        conf = raw.where((raw != 0) & (raw == raw.shift(1)), 0)
        confirmed = conf.values.astype(int)

        n = len(close)
        signal = np.zeros(n, dtype=int)

        # Path-dependent exit loop: profit-target OR time-stop
        position = 0
        entry_price = 0.0
        bars_held = 0
        pt = params.profit_target_pct / 100.0

        for i in range(1, n):
            if position != 0:
                bars_held += 1
                pnl = (close[i] - entry_price) / entry_price * position
                if pnl >= pt or bars_held >= params.time_stop_bars:
                    position = 0
                    entry_price = 0.0
                    bars_held = 0
                    # signal[i] remains 0 (exit)
                else:
                    signal[i] = position

            if position == 0 and confirmed[i] != 0:
                position = confirmed[i]
                entry_price = close[i]
                bars_held = 0
                signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
