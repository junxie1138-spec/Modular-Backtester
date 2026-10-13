from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AcceptanceParams:
    variance_window: int = 40
    atr_window: int = 14
    asymmetry: float = 1.6
    streak_min: int = 2
    rejection_location: float = 0.65
    breakeven_pct: float = 0.025
    trail_atr: float = 2.8
    max_hold: int = 18
    cooldown_bars: int = 3


class GeneratedStrategy(BaseStrategy[AcceptanceParams]):
    strategy_id = 'gen_local-w2_1790996218'

    @classmethod
    def params_type(cls) -> type[AcceptanceParams]:
        return AcceptanceParams

    def warmup_bars(self, params: AcceptanceParams) -> int:
        return int(max(params.variance_window + 1, params.atr_window + 1, params.streak_min + 1))

    def indicators(self, data: pd.DataFrame, params: AcceptanceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        returns = data['close'].pct_change(fill_method=None)
        squared = returns.pow(2)
        upside = squared.where(returns > 0, 0.0).where(returns.notna())
        downside = squared.where(returns < 0, 0.0).where(returns.notna())
        out['up_variance'] = upside.rolling(params.variance_window, min_periods=params.variance_window).mean()
        out['down_variance'] = downside.rolling(params.variance_window, min_periods=params.variance_window).mean()
        out['variance_ratio'] = (out['down_variance'] + 1e-12) / (out['up_variance'] + 1e-12)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        day_range = (data['high'] - data['low']).replace(0.0, np.nan)
        out['close_location'] = ((data['close'] - data['low']) / day_range).fillna(0.5)
        out['prior_midpoint'] = ((data['high'] + data['low']) / 2.0).shift(1)
        conditions = {
            'accept_up': (data['close'] > out['prior_midpoint']) & (data['close'] > data['open']),
            'accept_down': (data['close'] < out['prior_midpoint']) & (data['close'] < data['open']),
            'reject_down': (data['low'] < data['low'].shift(1)) & (out['close_location'] >= params.rejection_location),
            'reject_up': (data['high'] > data['high'].shift(1)) & (out['close_location'] <= 1.0 - params.rejection_location),
        }
        for name, condition in conditions.items():
            flag = condition.fillna(False).astype(bool)
            out[name + '_streak'] = flag.astype(int).groupby((~flag).cumsum()).cumsum()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AcceptanceParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.ones(count, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        highs = data['high'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        ratios = indicators['variance_ratio'].to_numpy(dtype=float)
        accept_up = indicators['accept_up_streak'].to_numpy(dtype=float)
        accept_down = indicators['accept_down_streak'].to_numpy(dtype=float)
        reject_down = indicators['reject_down_streak'].to_numpy(dtype=float)
        reject_up = indicators['reject_up_streak'].to_numpy(dtype=float)
        position = 0
        pending_entry = False
        entry = 0.0
        entry_atr = 0.0
        stop = 0.0
        armed = False
        held = 0
        position_size = 1.0
        next_entry = 0
        for i in range(count):
            if position != 0:
                if pending_entry:
                    # The preceding close's decision fills at this bar's open.
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        position = 0
                        pending_entry = False
                        next_entry = i + params.cooldown_bars + 1
                        continue
                    # Signed-price coordinates make the protective stop increase
                    # monotonically for both longs and shorts.
                    stop = position * entry - params.trail_atr * entry_atr
                    pending_entry = False
                    held = 0
                    armed = False
                held += 1
                adverse = lows[i] if position == 1 else -highs[i]
                signed_close = position * closes[i]
                stop_hit = np.isfinite(adverse) and adverse <= stop
                if stop_hit or held >= params.max_hold:
                    position = 0
                    next_entry = i + params.cooldown_bars + 1
                    continue
                # Test the previously established stop before updating it.
                # Close-based updates become effective on subsequent bars.
                if np.isfinite(signed_close):
                    if signed_close >= position * entry + entry * params.breakeven_pct:
                        armed = True
                    if armed:
                        stop = max(stop, position * entry)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, signed_close - params.trail_atr * atrs[i])
                signals[i] = position
                sizes[i] = position_size
                continue
            if i < next_entry:
                continue
            if not np.isfinite(ratios[i]) or not np.isfinite(atrs[i]) or atrs[i] <= 0:
                continue
            if not np.isfinite(closes[i]) or closes[i] <= 0:
                continue
            candidate = 0
            strength = 0.0
            if ratios[i] > params.asymmetry:
                if reject_down[i] >= params.streak_min:
                    candidate = 1
                    strength = reject_down[i]
            elif ratios[i] < 1.0 / params.asymmetry:
                if reject_up[i] >= params.streak_min:
                    candidate = -1
                    strength = reject_up[i]
            else:
                if accept_up[i] >= params.streak_min:
                    candidate = 1
                    strength = accept_up[i]
                elif accept_down[i] >= params.streak_min:
                    candidate = -1
                    strength = accept_down[i]
            if candidate != 0:
                position = candidate
                pending_entry = True
                entry_atr = atrs[i]
                position_size = float(0.35 + 0.60 * np.clip((strength - params.streak_min + 1.0) / 4.0, 0.0, 1.0))
                signals[i] = position
                sizes[i] = position_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
