from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    lookback_high: int = 20
    sickness_depth_pct: float = 0.02
    sickness_streak_min: int = 5
    healing_streak_required: int = 2
    ma_regime: int = 200
    profit_target_pct: float = 0.04
    time_stop_bars: int = 10


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1779646593"

    @classmethod
    def params_type(cls):
        return GenA1Params

    def warmup_bars(self, params: GenA1Params) -> int:
        return int(max(params.ma_regime, params.lookback_high)) + 2

    def indicators(self, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        close = data["close"].astype(float)

        lookback = max(int(params.lookback_high), 2)
        ma_window = max(int(params.ma_regime), 2)
        depth = float(params.sickness_depth_pct)

        rolling_high = close.rolling(lookback, min_periods=lookback).max()
        drawdown = (close / rolling_high) - 1.0

        sick_mask = (drawdown <= -depth)
        sick_mask = sick_mask.where(~drawdown.isna(), other=False).fillna(False)
        sick_int = sick_mask.astype(int)

        # Consecutive-streak count of sickness (resets on any non-sick bar).
        reset_groups_sick = (sick_int == 0).cumsum()
        sick_streak = sick_int.groupby(reset_groups_sick).cumsum().fillna(0).astype(int)

        # Healing bar: an up-close while still inside drawdown (sick=True).
        up_bar = (close.diff() > 0).fillna(False)
        heal_mask = (up_bar & sick_mask).fillna(False)
        heal_int = heal_mask.astype(int)
        reset_groups_heal = (heal_int == 0).cumsum()
        heal_streak = heal_int.groupby(reset_groups_heal).cumsum().fillna(0).astype(int)

        ma = close.rolling(ma_window, min_periods=ma_window).mean()
        above_ma = (close > ma).fillna(False).astype(int)

        out = pd.DataFrame(
            {
                "drawdown": drawdown,
                "sick_streak": sick_streak,
                "heal_streak": heal_streak,
                "above_ma": above_ma,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        idx = data.index
        n = len(data)

        sick_streak = indicators["sick_streak"].to_numpy(dtype=np.int64, na_value=0)
        heal_streak = indicators["heal_streak"].to_numpy(dtype=np.int64, na_value=0)
        above_ma = indicators["above_ma"].to_numpy(dtype=np.int64, na_value=0)
        close = data["close"].to_numpy(dtype=np.float64)

        sick_min = max(int(params.sickness_streak_min), 1)
        heal_req = max(int(params.healing_streak_required), 1)
        pt = float(params.profit_target_pct)
        time_stop = max(int(params.time_stop_bars), 1)

        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=np.float64)

        in_position = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            px = close[i]
            if not np.isfinite(px):
                signal[i] = 1 if in_position else 0
                if in_position:
                    bars_held += 1
                continue

            if in_position:
                bars_held += 1
                ret = (px / entry_price) - 1.0 if entry_price > 0 else 0.0
                if ret >= pt or bars_held >= time_stop:
                    signal[i] = 0
                    in_position = False
                    entry_price = 0.0
                    bars_held = 0
                else:
                    signal[i] = 1
            else:
                if (
                    above_ma[i] == 1
                    and sick_streak[i] >= sick_min
                    and heal_streak[i] >= heal_req
                ):
                    signal[i] = 1
                    in_position = True
                    entry_price = px
                    bars_held = 0

        df = pd.DataFrame({"signal": signal, "size": size}, index=idx)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
