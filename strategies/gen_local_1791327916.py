from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ImpactParams:
    rank_window: int = 60
    volume_window: int = 20
    atr_window: int = 14
    percentile: float = 0.90
    breakeven_pct: float = 0.006
    trail_atr: float = 1.25
    max_hold: int = 2

    def __post_init__(self):
        if min(self.rank_window, self.volume_window, self.atr_window) < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not 0.5 < self.percentile <= 1.0:
            raise ValueError('percentile must be in (0.5, 1].')
        if not np.isfinite(self.breakeven_pct) or self.breakeven_pct <= 0:
            raise ValueError('breakeven_pct must be finite and positive.')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive.')
        if self.max_hold not in (1, 2):
            raise ValueError('max_hold must be one or two sessions.')


class GeneratedStrategy(BaseStrategy[ImpactParams]):
    strategy_id = 'gen_local_1791327916'

    @classmethod
    def params_type(cls):
        return ImpactParams

    def warmup_bars(self, params: ImpactParams) -> int:
        return int(max(200, params.volume_window + params.rank_window - 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ImpactParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['mean_volume'] = data['volume'].rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean()
        out['relative_volume'] = data['volume'].where(data['volume'] > 0).div(
            out['mean_volume'].where(out['mean_volume'] > 0)
        )
        out['body_return'] = data['close'].div(
            data['open'].where(data['open'] > 0)
        ).sub(1.0)
        out['impact'] = out['body_return'].div(out['relative_volume'])
        out['absolute_impact'] = out['impact'].abs()
        out['impact_percentile'] = out['absolute_impact'].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'].sub(data['low']),
            data['high'].sub(out['previous_close']).abs(),
            data['low'].sub(out['previous_close']).abs()
        ], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out['entry_atr'] = out['atr'].shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ImpactParams) -> SignalFrame:
        required = ['ma200', 'relative_volume', 'impact',
                    'impact_percentile', 'atr']
        valid = indicators[required].notna().all(axis=1)
        valid &= (indicators['atr'] > 0) & (data['close'] > 0)
        extreme = valid & (indicators['impact_percentile'] >= params.percentile)
        extreme &= indicators['relative_volume'] < 1.0
        candidates = np.zeros(len(data), dtype=int)
        candidates[(extreme & (indicators['impact'] > 0)
                    & (data['close'] > indicators['ma200'])).to_numpy()] = 1
        candidates[(extreme & (indicators['impact'] < 0)
                    & (data['close'] < indicators['ma200'])).to_numpy()] = -1

        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entry_atrs = indicators['entry_atr'].to_numpy(dtype=float)
        averages = indicators['ma200'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        pending = False
        entry = 0.0
        stop = 0.0
        armed = False
        age = 0

        for i in range(len(data)):
            if side:
                if pending:
                    # The preceding close's decision fills at this bar's open.
                    if not (np.isfinite(opens[i]) and opens[i] > 0
                            and np.isfinite(entry_atrs[i]) and entry_atrs[i] > 0):
                        side = 0
                        pending = False
                        continue
                    entry = opens[i]
                    stop = side * entry - params.trail_atr * entry_atrs[i]
                    pending = False
                    armed = False
                    age = 0
                age += 1
                if not (np.isfinite(closes[i]) and closes[i] > 0
                        and np.isfinite(atrs[i]) and atrs[i] > 0
                        and np.isfinite(averages[i])):
                    side = 0
                    continue
                signed_close = side * closes[i]
                regime_ok = side * (closes[i] - averages[i]) > 0
                # Stops are evaluated at the close; exits fill next open.
                if signed_close <= stop or age >= params.max_hold or not regime_ok:
                    side = 0
                    continue
                if side * (closes[i] / entry - 1.0) >= params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed stops only rise: short price stops only fall.
                    stop = max(stop, side * entry,
                               signed_close - params.trail_atr * atrs[i])
                signals[i] = side
            elif candidates[i]:
                side = int(candidates[i])
                pending = True
                signals[i] = side

        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = ImpactParams()
    assert strategy.warmup_bars(params) >= 200
    for side in (1, -1):
        index = pd.date_range('2020-01-01', periods=4)
        close = np.array([100.0, 100.0 + side, 100.0, 100.0])
        data = pd.DataFrame({
            'open': 100.0, 'high': np.maximum(close, 100.0) + 1.0,
            'low': np.minimum(close, 100.0) - 1.0,
            'close': close, 'volume': 100.0
        }, index=index)
        ind = pd.DataFrame({
            'ma200': 100.0 - 10.0 * side, 'relative_volume': 0.5,
            'impact': [float(side), 0.0, 0.0, 0.0],
            'impact_percentile': [1.0, 0.5, 0.5, 0.5],
            'atr': 2.0, 'entry_atr': 2.0
        }, index=index)
        result = strategy.generate_signals(data, ind, None, params).data
        assert result['signal'].tolist() == [0, side, side, 0]
        assert result['size'].gt(0).all()
        # A one-session horizon permits only the entry session's exposure.
        short_hold = ImpactParams(max_hold=1)
        result = strategy.generate_signals(data, ind, None, short_hold).data
        assert result['signal'].tolist() == [0, side, 0, 0]
        # Missing ranks must never create an entry.
        ind['impact_percentile'] = np.nan
        result = strategy.generate_signals(data, ind, None, params).data
        assert result['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
