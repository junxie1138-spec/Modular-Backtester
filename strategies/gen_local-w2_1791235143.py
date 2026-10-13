from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    baseline_bars: int = 20
    absorption_bars: int = 3
    volume_multiple: float = 1.4
    impact_jump: float = 1.8
    absorption_ratio: float = 0.7
    atr_bars: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    initial_stop_atr: float = 2.0
    hold_bars: int = 4

    def __post_init__(self):
        if self.baseline_bars < 2 or self.absorption_bars < 1 or self.atr_bars < 2:
            raise ValueError('Lookbacks must be positive and baseline/ATR at least two.')
        if not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five.')
        values = (self.volume_multiple, self.impact_jump, self.absorption_ratio,
                  self.breakeven_pct, self.trail_atr, self.initial_stop_atr)
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Thresholds must be finite and positive.')
        if self.volume_multiple <= 1 or self.impact_jump <= 1 or self.absorption_ratio >= 1:
            raise ValueError('Require volume/impact expansion and prior absorption.')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1791235143'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params):
        return int(max(2 * params.baseline_bars + params.absorption_bars + 1,
                       params.atr_bars + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] > 0)
        out['return'] = close.pct_change(fill_method=None)
        out['volume_reference'] = volume.rolling(params.baseline_bars).median().shift(1)
        out['relative_volume'] = volume / out['volume_reference']
        out['impact'] = out['return'].abs() / out['relative_volume']
        out['impact_baseline'] = out['impact'].rolling(params.baseline_bars).median().shift(params.absorption_bars + 1)
        out['absorbed_impact'] = out['impact'].rolling(params.absorption_bars).mean().shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - close.shift(1)).abs(),
            (data['low'] - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        direction = np.sign(out['return'].fillna(0)).astype(int)
        volume_move = out['relative_volume'].ge(params.volume_multiple)
        repricing = (
            out['impact_baseline'].gt(0)
            & out['absorbed_impact'].le(params.absorption_ratio * out['impact_baseline'])
            & out['impact'].ge(params.impact_jump * out['impact_baseline'])
        )
        out['volume_move'] = direction.where(volume_move, 0)
        out['impact_change'] = direction.where(repricing, 0)
        out['candidate'] = out['volume_move'].where(
            out['volume_move'].eq(out['impact_change']) & out['atr'].gt(0), 0
        ).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        signals = np.zeros(len(data), dtype=int)
        side = 0
        pending = False
        entry = best = stop = 0.0
        armed = False
        age = 0
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        for i in range(len(data)):
            price, atr = closes[i], atrs[i]
            valid = np.isfinite(price) and price > 0 and np.isfinite(atr) and atr > 0
            if side:
                if pending:
                    # The previously emitted entry fills at this bar's open.
                    entry = opens[i]
                    prior_atr = atrs[i - 1]
                    valid = valid and np.isfinite(entry) and entry > 0 and np.isfinite(prior_atr) and prior_atr > 0
                    best = side * entry
                    stop = best - params.initial_stop_atr * prior_atr
                    pending = False
                age += 1
                signed_price = side * price
                exit_now = not valid or signed_price <= stop or age >= params.hold_bars
                if not exit_now:
                    # Activation, trailing and stop breaches are close-confirmed.
                    best = max(best, signed_price)
                    armed = armed or best - side * entry >= entry * params.breakeven_pct
                    if armed:
                        # The signed stop only increases: long stops rise, short stops fall.
                        stop = max(stop, side * entry, best - params.trail_atr * atr)
                        exit_now = signed_price <= stop
                if exit_now:
                    side = 0
            elif valid and candidates[i] != 0:
                side = int(candidates[i])
                pending = True
                armed = False
                age = 0
            signals[i] = side
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=6)
    for side in (1, -1):
        close = np.array([100., 100. + 2 * side, 100., 100., 100., 100.])
        data = pd.DataFrame({'open': 100., 'high': close + 1.,
                             'low': close - 1., 'close': close, 'volume': 1000.}, index=index)
        indicators = pd.DataFrame({'atr': 1., 'candidate': [side, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, side, side, 0, 0, 0]
        data['close'] = 100.
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, side, side, side, side, 0]
        assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
