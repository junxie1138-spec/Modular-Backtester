from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PaydayParams:
    volume_multiple: float = 1.15
    trail_atr: float = 1.5

    def __post_init__(self):
        if not np.isfinite(self.volume_multiple) or self.volume_multiple <= 0:
            raise ValueError('volume_multiple must be finite and positive')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[PaydayParams]):
    strategy_id = 'gen_local_1791339653'

    @classmethod
    def params_type(cls):
        return PaydayParams

    def warmup_bars(self, params: PaydayParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: PaydayParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = true_range.rolling(14, min_periods=14).mean()
        out['entry_atr'] = out['atr'].shift(1)
        baseline = data['volume'].rolling(20, min_periods=20).mean().shift(1)
        out['relative_volume'] = (data['volume'] / baseline.where(baseline > 0)).replace([np.inf, -np.inf], np.nan)
        candle_range = data['high'] - data['low']
        out['session_efficiency'] = ((data['close'] - data['open']) / candle_range.where(candle_range > 0)).replace([np.inf, -np.inf], np.nan)
        dates = data.index.to_series()
        previous_date = dates.shift(1)
        out['payday_crossing'] = (
            dates.dt.day.between(15, 19)
            & (previous_date.dt.day < 15)
            & (dates.dt.month == previous_date.dt.month)
            & (dates.dt.year == previous_date.dt.year)
        ).fillna(False)
        out['entry'] = (
            out['payday_crossing']
            & (out['relative_volume'] >= params.volume_multiple)
            & (out['session_efficiency'] >= 0.6)
            & (data['close'] > previous_close)
            & (out['atr'] > 0)
            & (out['entry_atr'] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: PaydayParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        highs = data['high'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entry_atrs = indicators['entry_atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        armed = False
        entry_price = 0.0
        peak = 0.0
        stop = 0.0
        age = 0

        for i in range(len(data)):
            # A preceding close decision becomes an entry at this bar's open.
            if not active and i > 0 and decisions[i - 1] == 1:
                if not (np.isfinite(opens[i]) and opens[i] > 0
                        and np.isfinite(entry_atrs[i]) and entry_atrs[i] > 0):
                    continue
                active = True
                armed = False
                entry_price = opens[i]
                peak = entry_price
                stop = entry_price - 2.0 * entry_atrs[i]
                age = 0

            if active:
                age += 1
                valid = (np.isfinite(highs[i]) and np.isfinite(lows[i])
                         and np.isfinite(closes[i]) and closes[i] > 0
                         and np.isfinite(atrs[i]) and atrs[i] > 0)
                if not valid or lows[i] <= stop:
                    active = False
                    continue

                peak = max(peak, highs[i])
                if peak >= entry_price * 1.01:
                    armed = True
                if armed:
                    stop = max(stop, entry_price, peak - params.trail_atr * atrs[i])

                # New high-based stops are checked at the close, avoiding
                # assumptions about the ordering of this bar's high and low.
                # Exit decisions execute next bar; stops are close decisions.
                if closes[i] <= stop or age >= 4:
                    active = False
                    continue
                decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1

        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2024-01-15', periods=8, freq='B')
    data = pd.DataFrame({
        'open': 100.0, 'high': 100.5, 'low': 99.5,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': 1.0, 'entry_atr': 1.0, 'entry': False,
    }, index=index)
    indicators.iloc[0, indicators.columns.get_loc('entry')] = True
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, PaydayParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[2], ['high', 'close']] = [102.0, 101.0]
    data.loc[index[3], ['open', 'high', 'low', 'close']] = [101.0, 101.2, 100.4, 100.8]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, PaydayParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
