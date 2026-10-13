from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RetestParams:
    entry_z: float = 1.7
    trail_atr: float = 0.8

    def __post_init__(self):
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError('entry_z must be finite and positive')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[RetestParams]):
    strategy_id = 'gen_local-w2_1790962894'

    @classmethod
    def params_type(cls):
        return RetestParams

    def warmup_bars(self, params: RetestParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: RetestParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['ma'] = close.rolling(20, min_periods=20).mean()
        out['std'] = close.rolling(20, min_periods=20).std(ddof=0)
        out['z'] = ((close - out['ma']) / out['std'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        out['tr'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        out['atr'] = out['tr'].rolling(14, min_periods=14).mean()
        z = out['z']
        spike = z.shift(2)
        pause = z.shift(1)
        retest = (
            (spike.abs() >= params.entry_z)
            & (pause * spike > 0)
            & (z * spike > 0)
            & (pause.abs() < spike.abs())
            & (z.abs() > pause.abs())
            & (z.abs() < spike.abs())
            & (z.abs() >= params.entry_z)
            & (out['atr'] > 0)
        )
        out['setup'] = np.where(retest, -np.sign(z.fillna(0)), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RetestParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        setups = indicators['setup'].to_numpy(dtype=int)
        side = 0
        entry = np.nan
        stop = np.nan
        armed = False
        age = 0
        for i in range(len(data)):
            if side:
                if np.isnan(entry):
                    if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                        side = 0
                        continue
                    entry = opens[i]
                    # Signed price makes both long and short stops ratchet upward.
                    stop = side * entry - atr[i - 1]
                    armed = False
                    age = 0
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                    side = 0
                    entry = np.nan
                    continue
                mark = side * closes[i]
                # Stops are close-confirmed; their exits execute next bar.
                if mark <= stop or age >= 2:
                    side = 0
                    entry = np.nan
                    continue
                if side * (closes[i] / entry - 1.0) >= 0.003:
                    armed = True
                if armed:
                    stop = max(stop, side * entry, mark - params.trail_atr * atr[i])
                raw[i] = side
            elif setups[i] and np.isfinite(opens[i]) and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(setups[i])
                entry = np.nan
                raw[i] = side
        df = pd.DataFrame({'signal': raw, 'size': np.ones(len(data), dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = RetestParams()
    index = pd.date_range('2020-01-01', periods=6)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=22)
    for side in (1, -1):
        data = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 98.0, 'close': 100.0, 'volume': 1000.0}, index=index)
        data.loc[index[3], 'close'] = 100.0 + side * 0.6
        indicators = pd.DataFrame({'atr': 1.0, 'setup': [0, 0, side, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result['signal'].tolist() == [0, 0, 0, side, side, 0]
        assert (result['size'] > 0).all()
        data.loc[index[3], 'close'] = 100.0 - side * 1.1
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result['signal'].tolist() == [0, 0, 0, side, 0, 0]
    flat = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=pd.date_range('2020-01-01', periods=30))
    assert not strategy.indicators(flat, params)['setup'].any()


if __name__ == '__main__':
    _self_check()
