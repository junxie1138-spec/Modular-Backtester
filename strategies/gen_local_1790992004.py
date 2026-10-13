from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    rank_window: int = 63
    history: int = 126
    shape_bars: int = 3
    hold_days: int = 4
    similarity_radius: float = 0.22
    min_analogs: int = 6
    forecast_floor: float = 0.002
    weak_gap_rank: float = 0.35
    strong_day_rank: float = 0.65
    atr_window: int = 14
    breakeven_gain: float = 0.008
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local_1790992004'

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(max(params.rank_window + params.shape_bars + params.history + params.hold_days, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        opening = data['open'].where(data['open'] > 0)
        out['gap_return'] = opening / close.shift(1) - 1.0
        out['day_return'] = close / opening - 1.0
        out['gap_rank'] = out['gap_return'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['day_rank'] = out['day_return'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        previous = close.shift(1)
        out['true_range'] = pd.concat([data['high'] - data['low'], (data['high'] - previous).abs(), (data['low'] - previous).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['horizon_return'] = close / close.shift(params.hold_days) - 1.0
        total_weight = pd.Series(0.0, index=data.index)
        total_payoff = pd.Series(0.0, index=data.index)
        analog_count = pd.Series(0, index=data.index)
        for lag in range(params.hold_days + 1, params.hold_days + params.history + 1):
            distance = pd.Series(0.0, index=data.index)
            for step in range(params.shape_bars):
                for column in ('gap_rank', 'day_rank'):
                    distance = distance + (out[column].shift(step) - out[column].shift(lag + step)).pow(2)
            distance = np.sqrt(distance / (2.0 * params.shape_bars))
            payoff = out['horizon_return'].shift(lag - params.hold_days)
            qualifies = distance.le(params.similarity_radius) & payoff.notna()
            weight = np.exp(-distance.pow(2) / max(params.similarity_radius ** 2, 1e-12)).where(qualifies, 0.0).fillna(0.0)
            total_weight = total_weight + weight
            total_payoff = total_payoff + weight * payoff.fillna(0.0)
            analog_count = analog_count + qualifies.astype(int)
        out['analog_count'] = analog_count
        out['analog_forecast'] = total_payoff / total_weight.replace(0.0, np.nan)
        out['analog_agreement'] = out['analog_count'].ge(params.min_analogs) & out['analog_forecast'].gt(params.forecast_floor)
        out['absorption_agreement'] = out['gap_rank'].le(params.weak_gap_rank) & out['day_rank'].ge(params.strong_day_rank) & out['day_return'].gt(0.0)
        out['entry'] = (out['analog_agreement'] & out['absorption_agreement'] & out['atr'].gt(0.0)).fillna(False)
        out.iloc[:self.warmup_bars(params), out.columns.get_loc('entry')] = False
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        count = len(data)
        desired = np.zeros(count, dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_price = np.nan
        entry_bar = -1
        stop = np.nan
        peak_close = np.nan
        armed = False
        holding_days = min(5, max(3, int(params.hold_days)))
        for i in range(count):
            if active:
                if entry_bar < 0:
                    entry_price = opens[i]
                    entry_bar = i
                    peak_close = entry_price
                    stop = entry_price - params.trail_atr * atrs[i - 1]
                valid = np.isfinite(entry_price) and entry_price > 0.0 and np.isfinite(lows[i]) and np.isfinite(closes[i]) and np.isfinite(atrs[i]) and np.isfinite(stop)
                if not valid or opens[i] <= stop or lows[i] <= stop or i - entry_bar + 1 >= holding_days:
                    active = False
                    entry_bar = -1
                    armed = False
                    continue
                peak_close = max(peak_close, closes[i])
                if closes[i] >= entry_price * (1.0 + params.breakeven_gain):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, peak_close - params.trail_atr * atrs[i])
                desired[i] = 1
            elif entries[i] and i + 1 < count:
                active = True
                entry_bar = -1
                armed = False
                desired[i] = 1
        df = pd.DataFrame({'signal': desired, 'size': np.ones(count, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
