from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ordinal_window: int = 24
    entropy_ceiling: float = 0.90
    entropy_drop: float = 0.06
    snr_window: int = 15
    snr_floor: float = 0.80
    atr_window: int = 14
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791309896'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.ordinal_window + 8, params.snr_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        a = out['return'].shift(2)
        b = out['return'].shift(1)
        c = out['return']
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & b.ne(c) & a.ne(c)
        code = a.gt(b).astype(int) + 2 * b.gt(c).astype(int) + 4 * a.gt(c).astype(int)
        out['ordinal_code'] = code.where(valid)
        out['valid_patterns'] = valid.astype(float).rolling(params.ordinal_window, min_periods=params.ordinal_window).sum()
        entropy = pd.Series(0.0, index=data.index)
        for pattern in (0, 1, 2, 5, 6, 7):
            probability = out['ordinal_code'].eq(pattern).astype(float).rolling(params.ordinal_window, min_periods=params.ordinal_window).mean()
            out[f'pattern_probability_{pattern}'] = probability
            entropy -= probability * np.log(probability.where(probability.gt(0), 1.0))
        out['entropy'] = (entropy / np.log(6.0)).where(out['valid_patterns'].eq(params.ordinal_window))
        out['entropy_change'] = out['entropy'].diff(5)
        out['return_sum'] = out['return'].rolling(params.snr_window, min_periods=params.snr_window).sum()
        out['return_energy'] = out['return'].pow(2).rolling(params.snr_window, min_periods=params.snr_window).sum()
        out['snr'] = out['return_sum'] / np.sqrt(out['return_energy'].where(out['return_energy'].gt(0)))
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([(data['high'] - data['low']).abs(), (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['complexity_agrees'] = out['entropy'].lt(params.entropy_ceiling) & out['entropy_change'].le(-params.entropy_drop)
        out['snr_agrees'] = out['snr'].gt(params.snr_floor)
        out['entry'] = (out['complexity_agrees'] & out['snr_agrees'] & out['atr'].gt(0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        high_water = np.nan
        stop = -np.inf
        held = 0
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if active:
                held += 1
                if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                    active = False
                else:
                    # The first update is on the actual entry bar, after the signal shift.
                    high_water = close if np.isnan(high_water) else max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                    if close <= stop or held >= params.hold_bars:
                        active = False
                signal[i] = int(active)
                # Never schedule a fresh entry on the same close as an exit.
                continue
            if entries[i] and np.isfinite(close) and np.isfinite(atr) and atr > 0:
                active = True
                high_water = np.nan
                stop = -np.inf
                held = 0
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = Params()
    index = pd.date_range('2020-01-01', periods=25)
    data = pd.DataFrame({'close': np.full(25, 100.0)}, index=index)
    indicators = pd.DataFrame({'atr': np.ones(25), 'entry': [True] + [False] * 24}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0] + [1] * 18 + [0] * 6
    data.loc[index[2], 'close'] = 110.0
    data.loc[index[3], 'close'] = 105.0
    indicators.loc[index[3], 'atr'] = 10.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[:5].tolist() == [0, 1, 1, 1, 0]
    assert result['size'].gt(0).all()
    assert set(result['signal'].unique()) <= {0, 1}


if __name__ == '__main__':
    _self_check()
