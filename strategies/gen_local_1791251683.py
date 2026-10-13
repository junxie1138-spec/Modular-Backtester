from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapChangeParams:
    baseline_bars: int = 40
    atr_bars: int = 14
    evidence_threshold: float = 3.0
    drift_allowance: float = 0.25
    trail_atr: float = 3.0
    holding_bars: int = 18

    def __post_init__(self):
        if self.baseline_bars < 2 or self.atr_bars < 1 or self.holding_bars < 1:
            raise ValueError('Lookbacks and holding period must be positive; baseline must be at least two.')
        if not all(np.isfinite(x) and x > 0 for x in (self.evidence_threshold, self.trail_atr)):
            raise ValueError('Evidence threshold and ATR multiplier must be finite and positive.')
        if not np.isfinite(self.drift_allowance) or self.drift_allowance < 0:
            raise ValueError('Drift allowance must be finite and nonnegative.')


class GeneratedStrategy(BaseStrategy[GapChangeParams]):
    strategy_id = 'gen_local_1791251683'

    @classmethod
    def params_type(cls):
        return GapChangeParams

    def warmup_bars(self, params: GapChangeParams) -> int:
        return int(max(params.baseline_bars + 3, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GapChangeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        previous = close.shift(1)
        out['gap'] = ((data['open'].where(data['open'] > 0) - previous) / previous).replace([np.inf, -np.inf], np.nan)
        history = out['gap'].shift(1)
        out['gap_mean'] = history.rolling(params.baseline_bars, min_periods=params.baseline_bars).mean()
        out['gap_std'] = history.rolling(params.baseline_bars, min_periods=params.baseline_bars).std(ddof=0)
        scale = out['gap_std'].where(out['gap_std'] > 1e-10)
        out['innovation'] = ((out['gap'] - out['gap_mean']) / scale).replace([np.inf, -np.inf], np.nan)
        out['up_sum'] = (out['innovation'] - params.drift_allowance).fillna(0.0).cumsum()
        out['down_sum'] = (-out['innovation'] - params.drift_allowance).fillna(0.0).cumsum()
        out['up_evidence'] = out['up_sum'] - out['up_sum'].cummin().clip(upper=0.0)
        out['down_evidence'] = out['down_sum'] - out['down_sum'].cummin().clip(upper=0.0)
        out['true_range'] = pd.concat([(data['high'] - data['low']), (data['high'] - previous).abs(), (data['low'] - previous).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out['close_change'] = close.diff()
        out['ready'] = out['innovation'].notna() & np.isfinite(close) & np.isfinite(out['atr']) & out['atr'].gt(0)
        bullish = out['up_evidence'].ge(params.evidence_threshold) & out['up_evidence'].gt(out['down_evidence']) & out['close_change'].gt(0) & out['ready']
        bearish = out['down_evidence'].ge(params.evidence_threshold) & out['down_evidence'].gt(out['up_evidence']) & out['close_change'].lt(0) & out['ready']
        out['candidate'] = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        out['confirmed'] = out['candidate'].where(out['candidate'].eq(out['candidate'].shift(1)), 0).astype(int)
        out['entry'] = out['confirmed'].where(out['confirmed'].ne(out['confirmed'].shift(1).fillna(0)), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapChangeParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        low_water = np.nan
        high_stop = -np.inf
        short_stop = np.inf
        for i in range(len(data)):
            price, atr = closes[i], atrs[i]
            valid = np.isfinite(price) and price > 0 and np.isfinite(atr) and atr > 0
            if position:
                age += 1
                if valid:
                    # Initialize watermarks on the fill bar, after the entry decision.
                    high_water = price if not np.isfinite(high_water) else max(high_water, price)
                    low_water = price if not np.isfinite(low_water) else min(low_water, price)
                    # The required highest-close stop ratchets upward for both directions.
                    high_stop = max(high_stop, high_water - params.trail_atr * atr)
                    # Shorts additionally have a mirrored adverse-move trailing stop.
                    short_stop = min(short_stop, low_water + params.trail_atr * atr)
                    stopped = price <= high_stop or (position == -1 and price >= short_stop)
                else:
                    stopped = True
                if stopped or age >= params.holding_bars:
                    position = 0
                signals[i] = position
                continue
            if valid and entries[i] != 0:
                position = int(entries[i])
                age = 0
                high_water = np.nan
                low_water = np.nan
                high_stop = -np.inf
                short_stop = np.inf
            signals[i] = position
        df['signal'] = signals
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = GapChangeParams(trail_atr=2.0)
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame({'close': [100., 101., 106., 105., 103., 102., 100., 99., 102.]}, index=index)
    features = pd.DataFrame({'atr': [1., 1., 1., 10., 10., 1., 1., 1., 1.], 'entry': [1, 0, 0, 0, 0, 0, -1, 0, 0]}, index=index)
    result = strategy.generate_signals(data, features, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, -1, -1]
    assert result['size'].gt(0).all()
    params = GapChangeParams(holding_bars=2, trail_atr=100.0)
    result = strategy.generate_signals(data, features, None, params).data
    assert result['signal'].iloc[:4].tolist() == [0, 1, 1, 0]
    prices = pd.Series(np.linspace(100.0, 120.0, 80), index=pd.date_range('2021-01-01', periods=80))
    bars = pd.DataFrame({'open': prices, 'high': prices + 1, 'low': prices - 1, 'close': prices, 'volume': 1000.0})
    derived = strategy.indicators(bars, GapChangeParams())
    assert derived.index.equals(bars.index)
    assert derived['entry'].iloc[:strategy.warmup_bars(GapChangeParams()) - 1].eq(0).all()
    assert not np.isinf(derived.select_dtypes(include=[np.number]).to_numpy()).any()


if __name__ == '__main__':
    _self_check()
