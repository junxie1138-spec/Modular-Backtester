from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    calibration_window: int = 126
    min_streak: int = 2
    min_observations: int = 8
    continuation_edge: float = 0.05
    atr_window: int = 14
    initial_stop_atr: float = 2.0
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = 'gen_local_1790995388'

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return int(max(params.calibration_window + 2, params.atr_window + 1, params.min_streak + 2))

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        change = data['close'].diff()
        up = change.gt(0)
        groups = (~up).cumsum()
        streak = up.astype(int).groupby(groups).cumsum()
        eligible = streak.shift(1).ge(params.min_streak)
        observations = eligible.astype(float).rolling(params.calibration_window, min_periods=params.calibration_window).sum().shift(1)
        wins = (eligible & up).astype(float).rolling(params.calibration_window, min_periods=params.calibration_window).sum().shift(1)
        baseline = up.astype(float).where(change.notna()).rolling(params.calibration_window, min_periods=params.calibration_window).mean().shift(1)
        probability = (wins + 1.0) / (observations + 2.0)
        premium = probability - baseline
        previous_close = data['close'].shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        qualifies = (
            streak.ge(params.min_streak)
            & observations.ge(params.min_observations)
            & premium.ge(params.continuation_edge)
            & atr.gt(0)
        ).fillna(False)
        out['change'] = change
        out['up'] = up.astype(int)
        out['streak'] = streak
        out['eligible'] = eligible.astype(int)
        out['observations'] = observations
        out['wins'] = wins
        out['baseline'] = baseline
        out['continuation_probability'] = probability
        out['continuation_premium'] = premium
        out['previous_close'] = previous_close
        out['true_range'] = true_range
        out['atr'] = atr
        out['qualifies'] = qualifies.astype(int)
        out['confirmed'] = (qualifies & qualifies.shift(1, fill_value=False)).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StreakParams) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        streaks = indicators['streak'].to_numpy(dtype=int)
        confirmed = indicators['confirmed'].to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        armed = True
        breakeven = False
        entry = 0.0
        stop = 0.0
        held = 0
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0:
                    entry = opens[i]
                    stop = entry - params.initial_stop_atr * atrs[i - 1]
                    held = 0
                    breakeven = False
                    active = True

            if active:
                held += 1
                valid = np.isfinite(lows[i]) and np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0
                # Test the previously established stop before raising it at this close.
                # Stop violations produce a flat target for the following bar.
                if not valid or lows[i] <= stop or held >= params.holding_bars:
                    active = False
                    signals[i] = 0
                    continue
                if closes[i] >= entry * (1.0 + params.breakeven_pct):
                    breakeven = True
                if breakeven:
                    stop = max(stop, entry, closes[i] - params.trail_atr * atrs[i])
                    if closes[i] <= stop:
                        active = False
                        signals[i] = 0
                        continue
                signals[i] = 1
                continue

            if streaks[i] == 0:
                armed = True
            if i >= warmup - 1 and armed and confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0:
                signals[i] = 1
                pending_entry = True
                armed = False

        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
