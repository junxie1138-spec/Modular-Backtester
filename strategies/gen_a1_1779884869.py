from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA11779884869Params:
    rank_window: int = 30
    streak_threshold: int = 4
    top_percentile: float = 0.85
    bottom_percentile: float = 0.15
    hold_bars: int = 18
    regime_ma: int = 200
    use_regime_filter: bool = True


class GeneratedStrategy(BaseStrategy[GenA11779884869Params]):
    strategy_id = "gen_a1_1779884869"

    @classmethod
    def params_type(cls):
        return GenA11779884869Params

    def warmup_bars(self, params: GenA11779884869Params) -> int:
        return int(max(params.rank_window, params.regime_ma)) + int(params.streak_threshold) + 5

    def indicators(self, data: pd.DataFrame, params: GenA11779884869Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)
        w = int(params.rank_window)
        top = float(params.top_percentile)
        bot = float(params.bottom_percentile)

        close_rank = close.rolling(w, min_periods=w).rank(pct=True)
        volume_rank = volume.rolling(w, min_periods=w).rank(pct=True)

        close_high = (close_rank >= top).fillna(False).astype(int)
        close_low = (close_rank <= bot).fillna(False).astype(int)
        volume_high = (volume_rank >= top).fillna(False).astype(int)
        volume_low = (volume_rank <= bot).fillna(False).astype(int)

        def streak(flag: pd.Series) -> pd.Series:
            f = flag.fillna(0).astype(int)
            grp = (f == 0).cumsum()
            return f.groupby(grp).cumsum().astype(int)

        predator_streak = streak(close_high)
        prey_low_streak = streak(volume_low)
        absent_streak = streak(close_low)
        prey_high_streak = streak(volume_high)

        ma_w = int(params.regime_ma)
        ma = close.rolling(ma_w, min_periods=ma_w).mean()

        out = pd.DataFrame(index=data.index)
        out["close_rank"] = close_rank
        out["volume_rank"] = volume_rank
        out["predator_streak"] = predator_streak
        out["prey_low_streak"] = prey_low_streak
        out["absent_streak"] = absent_streak
        out["prey_high_streak"] = prey_high_streak
        out["ma"] = ma
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA11779884869Params,
    ) -> SignalFrame:
        n = len(data)
        threshold = int(params.streak_threshold)
        hold = int(params.hold_bars)
        if hold < 1:
            hold = 1
        use_regime = bool(params.use_regime_filter)

        predator_streak = indicators["predator_streak"].fillna(0).to_numpy(dtype=np.int64)
        prey_low_streak = indicators["prey_low_streak"].fillna(0).to_numpy(dtype=np.int64)
        absent_streak = indicators["absent_streak"].fillna(0).to_numpy(dtype=np.int64)
        prey_high_streak = indicators["prey_high_streak"].fillna(0).to_numpy(dtype=np.int64)
        close = data["close"].astype(float).to_numpy()
        ma = indicators["ma"].to_numpy(dtype=float)

        signals = np.zeros(n, dtype=np.int64)
        position = 0
        countdown = 0

        for i in range(n):
            if position != 0:
                signals[i] = position
                countdown -= 1
                if countdown <= 0:
                    position = 0
                continue

            short_trigger = (
                predator_streak[i] >= threshold
                and prey_low_streak[i] >= threshold
            )
            long_trigger = (
                absent_streak[i] >= threshold
                and prey_high_streak[i] >= threshold
            )

            if not (short_trigger or long_trigger):
                continue

            ma_i = ma[i]
            if use_regime:
                if np.isnan(ma_i):
                    continue
                above = close[i] > ma_i
                if short_trigger and above:
                    position = -1
                    countdown = hold
                    signals[i] = -1
                elif long_trigger and not above:
                    position = 1
                    countdown = hold
                    signals[i] = 1
            else:
                if short_trigger:
                    position = -1
                    countdown = hold
                    signals[i] = -1
                elif long_trigger:
                    position = 1
                    countdown = hold
                    signals[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signals, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
