from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PrevalenceParams:
    rank_window: int = 126
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[PrevalenceParams]):
    strategy_id = 'gen_local_1791306326'

    @classmethod
    def params_type(cls):
        return PrevalenceParams

    @staticmethod
    def warmup_bars(params: PrevalenceParams) -> int:
        if params.rank_window < 2:
            raise ValueError('rank_window must be at least 2')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive')
        return int(max(15, params.rank_window + 11))

    @staticmethod
    def indicators(data: pd.DataFrame, params: PrevalenceParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['change'] = close.diff()
        out['advancing'] = (out['change'] > 0).astype(float).where(out['change'].notna())
        out['prevalence'] = out['advancing'].rolling(10, min_periods=10).mean()
        out['percentile'] = out['prevalence'].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method='average', pct=True)
        out['previous_percentile'] = out['percentile'].shift(1)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['return_5'] = close / close.shift(5) - 1.0
        out['recovery'] = (
            (out['previous_percentile'] <= 0.25)
            & (out['percentile'] > 0.25)
            & (out['prevalence'] >= 0.4)
        )
        out['expansion'] = (
            (out['previous_percentile'] <= 0.65)
            & (out['percentile'] > 0.65)
            & (out['prevalence'] >= 0.6)
            & (out['prevalence'] < 0.8)
        )
        out['ready'] = np.isfinite(out[[
            'prevalence', 'percentile', 'previous_percentile', 'atr', 'return_5'
        ]]).all(axis=1) & (out['atr'] > 0)
        out['setup'] = (
            out['ready']
            & (out['recovery'] | out['expansion'])
            & (out['return_5'] > 0)
            & (close > data['open'])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PrevalenceParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        setups = indicators['setup'].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry = 0.0
        stop = 0.0
        peak = 0.0
        age = 0

        for i in range(len(data)):
            # A prior close's entry decision fills at this bar's open.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry = opens[i]
                    peak = entry
                    stop = entry - 3.0 * atrs[i - 1]
                    armed = False
                    age = 0

            if active:
                age += 1
                valid = (
                    np.isfinite(lows[i]) and np.isfinite(closes[i])
                    and np.isfinite(atrs[i]) and closes[i] > 0
                    and atrs[i] > 0
                )
                # Test the previously established stop before raising it.
                # Stop breaches request a next-open exit through the shifted signal.
                if not valid or lows[i] <= stop or age >= 20:
                    active = False
                    continue
                peak = max(peak, closes[i])
                if closes[i] >= entry * 1.025:
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atrs[i])
                    if closes[i] <= stop:
                        active = False
                        continue
                decisions[i] = 1
            elif setups[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = True
                decisions[i] = 1

        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = PrevalenceParams()
    index = pd.date_range('2020-01-01', periods=25)
    data = pd.DataFrame({
        'open': 100.0, 'high': 104.0, 'low': 100.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({'atr': 2.0, 'setup': False}, index=index)
    indicators.loc[index[0], 'setup'] = True
    data.loc[index[2:4], 'close'] = 103.0
    data.loc[index[4], 'low'] = 99.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[:6].tolist() == [0, 1, 1, 1, 1, 0]
    prefix = GeneratedStrategy.generate_signals(
        data.iloc[:4], indicators.iloc[:4], None, params
    ).data
    assert prefix['signal'].equals(result['signal'].iloc[:4])
    data['close'] = 100.0
    data['low'] = 100.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[1:21].eq(1).all()
    assert result['signal'].iloc[21:].eq(0).all()
    assert result['size'].gt(0).all()
    derived = GeneratedStrategy.indicators(data, params)
    assert not derived['setup'].any()
    assert GeneratedStrategy.warmup_bars(params) == 137


if __name__ == '__main__':
    _self_check()
