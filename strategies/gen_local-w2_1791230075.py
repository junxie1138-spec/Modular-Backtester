from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    volume_window: int = 20
    concordance_window: int = 10
    volatility_window: int = 20
    atr_window: int = 14
    correlation_threshold: float = 0.25
    volume_confirmation: float = 1.2
    target_volatility: float = 0.12
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    max_hold_bars: int = 10

    def __post_init__(self):
        windows = (self.volume_window, self.concordance_window,
                   self.volatility_window, self.atr_window)
        if any(w < 2 for w in windows) or self.max_hold_bars < 1:
            raise ValueError('Windows must be >= 2 and holding period >= 1')
        values = (self.correlation_threshold, self.volume_confirmation,
                  self.target_volatility, self.breakeven_pct, self.trail_atr)
        if not all(np.isfinite(v) for v in values):
            raise ValueError('Parameters must be finite')
        if not 0 < self.correlation_threshold < 1:
            raise ValueError('Correlation threshold must be between 0 and 1')
        if min(values[1:]) <= 0:
            raise ValueError('Volume, volatility and stop parameters must be positive')


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = 'gen_local-w2_1791230075'

    @classmethod
    def params_type(cls):
        return ParticipationParams

    def warmup_bars(self, params: ParticipationParams) -> int:
        return int(max(200,
                       params.volume_window + params.concordance_window + 1,
                       params.volatility_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ParticipationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['return'] = close.pct_change(fill_method=None)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['volume_baseline'] = data['volume'].shift(1).rolling(
            params.volume_window, min_periods=params.volume_window).mean()
        out['relative_volume'] = data['volume'].div(
            out['volume_baseline'].where(out['volume_baseline'] > 0))
        out['concordance'] = out['return'].rolling(
            params.concordance_window, min_periods=params.concordance_window
        ).corr(out['relative_volume']).clip(-1.0, 1.0)
        out['drift'] = out['return'].rolling(
            params.concordance_window, min_periods=params.concordance_window).sum()
        out['ann_vol'] = out['return'].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0) * np.sqrt(252.0)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[['ma200', 'concordance', 'relative_volume',
                     'drift', 'ann_vol', 'atr']].notna().all(axis=1)
        out['entry'] = (
            ready & (close > out['ma200']) & (out['drift'] > 0)
            & (out['return'] > 0)
            & (out['relative_volume'] >= params.volume_confirmation)
            & (out['concordance'] >= params.correlation_threshold)
            & (out['concordance'].shift(1).fillna(0.0) < params.correlation_threshold)
            & (out['ann_vol'] > 0) & (out['atr'] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ParticipationParams) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.05, dtype=float)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ma = indicators['ma200'].to_numpy(dtype=float)
        volatility = indicators['ann_vol'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        valid = (np.isfinite(prices).all(axis=1)
                 & (prices > 0).all(axis=1)
                 & np.isfinite(atr) & (atr > 0)
                 & np.isfinite(ma)
                 & np.isfinite(volatility) & (volatility > 0))
        active = False
        pending = False
        armed = False
        entry_price = 0.0
        stop = 0.0
        held_size = 0.05
        age = 0
        for i in range(n):
            if not valid[i]:
                active = pending = armed = False
                continue
            opening, high, low, close = prices[i]
            if pending:
                # The previous close's decision fills at this bar's open.
                entry_price = opening
                stop = entry_price - params.trail_atr * atr[i - 1]
                active = True
                pending = False
                armed = False
                age = 0
            if active:
                age += 1
                # Test the previously established stop before raising it.
                # Exit decisions execute next bar through the mandatory shift.
                if low <= stop or close <= ma[i] or age >= params.max_hold_bars:
                    active = armed = False
                    continue
                if high >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, high - params.trail_atr * atr[i])
                signal[i] = 1
                size[i] = held_size
                continue
            if entries[i]:
                held_size = float(np.clip(
                    params.target_volatility / volatility[i], 0.05, 0.95))
                pending = True
                signal[i] = 1
                size[i] = held_size
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({
        'open': [100.0] * 5,
        'high': [101.0, 104.0, 103.0, 103.0, 101.0],
        'low': [99.0, 99.0, 102.0, 101.0, 99.0],
        'close': [100.0, 103.0, 102.5, 102.0, 100.0],
        'volume': [1000.0] * 5
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0, 1.0, 4.0, 1.0, 1.0],
        'ma200': [90.0] * 5,
        'ann_vol': [0.2] * 5,
        'entry': [True, False, False, False, False]
    }, index=index)
    result = GeneratedStrategy().generate_signals(
        data, indicators, None, ParticipationParams()).data
    # ATR expansion cannot lower the 101.5 stop established on bar 1.
    # The bar-3 breach exits on bar 4; bar-1's low predates its raised stop.
    assert result['signal'].tolist() == [0, 1, 1, 1, 0]
    assert np.isfinite(result['size']).all() and (result['size'] > 0).all()
    assert abs(result['size'].iloc[1] - 0.6) < 1e-12


if __name__ == '__main__':
    _self_check()
