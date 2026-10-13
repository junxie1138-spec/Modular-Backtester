from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 12
    correlation_window: int = 30
    area_ceiling: float = 0.14
    correlation_floor: float = 0.10
    atr_window: int = 14
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1790964664'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        if min(params.signature_window, params.correlation_window) < 3:
            raise ValueError('Signature and correlation windows must be at least three.')
        if params.atr_window < 1 or params.max_hold < 1:
            raise ValueError('ATR window and holding limit must be positive.')
        if not all(np.isfinite(v) for v in (params.area_ceiling, params.correlation_floor, params.breakeven_pct, params.trail_atr)):
            raise ValueError('Thresholds must be finite.')
        if params.area_ceiling <= 0 or params.breakeven_pct <= 0 or params.trail_atr <= 0:
            raise ValueError('Area ceiling, breakeven threshold and ATR multiple must be positive.')
        if not -1 <= params.correlation_floor <= 1:
            raise ValueError('Correlation floor must lie between minus one and one.')
        return int(max(params.signature_window + 3, params.correlation_window + 3, params.atr_window + 2))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = np.log(close).diff()
        out['lag_return'] = out['return'].shift(1)
        out['lag2_return'] = out['return'].shift(2)
        w = params.signature_window
        # Antisymmetric level-two signature of the delay path, closed by its chord.
        out['edge_cross'] = out['lag_return'].pow(2) - out['lag2_return'] * out['return']
        out['chord_cross'] = out['return'] * out['return'].shift(w) - out['lag_return'] * out['return'].shift(w - 1)
        out['signed_area'] = 0.5 * (out['edge_cross'].rolling(w - 1).sum() + out['chord_cross'])
        out['energy'] = (out['return'].pow(2) + out['lag_return'].pow(2)).rolling(w).sum()
        out['area_ratio'] = out['signed_area'].abs().div(out['energy'].where(out['energy'] > 0))
        out['autocorrelation'] = out['return'].rolling(params.correlation_window).corr(out['lag_return'])
        out['drift'] = out['return'].rolling(w).sum()
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['setup'] = (
            out['area_ratio'].lt(params.area_ceiling)
            & out['autocorrelation'].gt(params.correlation_floor)
            & out['drift'].gt(0)
            & out['atr'].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        self.warmup_bars(params)
        setup = indicators['setup'].fillna(False).astype(bool)
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy()
        bars = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = stop = peak = 0.0
        entry_bar = 0
        for i, (open_px, high, low, close) in enumerate(bars):
            if not np.isfinite([open_px, high, low, close, atr[i]]).all() or min(open_px, low, close, atr[i]) <= 0:
                active = pending = False
                continue
            if pending:
                # The prior close's decision becomes a position at this open.
                entry = open_px
                peak = entry
                stop = entry - params.trail_atr * atr[i - 1]
                entry_bar = i
                armed = False
                active = True
                pending = False
            if active:
                # Test the existing stop before incorporating this bar's high.
                exit_now = low <= stop
                peak = max(peak, high)
                if high >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                # A new stop can also be breached by the observed closing price.
                exit_now = exit_now or close <= stop or i - entry_bar + 1 >= params.max_hold
                if exit_now:
                    active = False
                else:
                    decisions[i] = 1
            elif confirmed[i]:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = SignatureParams(trail_atr=2.0)
    index = pd.date_range('2020-01-01', periods=8)
    close = np.array([100, 100, 100, 103, 103, 101, 100, 100], dtype=float)
    data = pd.DataFrame({'open': 100.0, 'high': close + 1, 'low': close - 0.5, 'close': close, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'setup': [True, True, False, False, False, False, False, False], 'atr': 1.0}, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    indicators.loc[index[1], 'setup'] = False
    assert strategy.generate_signals(data, indicators, ctx, params).data['signal'].eq(0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index) and not derived['setup'].any()
