from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTransportParams:
    transport_window: int = 5
    baseline_window: int = 30
    atr_window: int = 14
    expansion_threshold: float = 1.05
    transport_threshold: float = 0.12
    target_volatility: float = 0.12
    breakeven_pct: float = 0.01
    trail_atr: float = 2.5
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[RangeTransportParams]):
    strategy_id = 'gen_local-w2_1791248510'

    @classmethod
    def params_type(cls) -> type[RangeTransportParams]:
        return RangeTransportParams

    def warmup_bars(self, params: RangeTransportParams) -> int:
        return int(max(params.baseline_window, params.transport_window, params.atr_window, 3) + 1)

    def indicators(self, data: pd.DataFrame, params: RangeTransportParams) -> pd.DataFrame:
        if min(params.transport_window, params.baseline_window, params.atr_window, params.holding_bars) < 1:
            raise ValueError('Lookbacks and holding_bars must be positive.')
        if not all(np.isfinite(v) and v > 0 for v in (
            params.expansion_threshold, params.transport_threshold,
            params.target_volatility, params.breakeven_pct, params.trail_atr
        )):
            raise ValueError('Thresholds and risk parameters must be finite and positive.')
        out = pd.DataFrame(index=data.index)
        high = data['high'].astype(float)
        low = data['low'].astype(float)
        close = data['close'].astype(float)
        valid = (high >= low) & (low > 0) & (close > 0)
        out['range'] = (high - low).where(valid)
        out['high_change'] = high.where(valid).diff()
        out['low_change'] = low.where(valid).diff()
        out['up_transport'] = pd.concat([
            out['high_change'].clip(lower=0), out['low_change'].clip(lower=0)
        ], axis=1).min(axis=1, skipna=False)
        out['down_transport'] = pd.concat([
            (-out['high_change']).clip(lower=0), (-out['low_change']).clip(lower=0)
        ], axis=1).min(axis=1, skipna=False)
        out['transport'] = out['up_transport'] - out['down_transport']
        out['transport_sum'] = out['transport'].rolling(params.transport_window).sum()
        out['range_sum'] = out['range'].rolling(params.transport_window).sum()
        out['score'] = (out['transport_sum'] / out['range_sum'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out['fast_range'] = out['range'].rolling(3).mean()
        out['baseline_range'] = out['range'].rolling(params.baseline_window).mean()
        out['expansion'] = out['fast_range'] / out['baseline_range'].replace(0, np.nan)
        out['previous_close'] = close.where(valid).shift(1)
        out['true_range'] = pd.concat([
            out['range'], (high - out['previous_close']).abs(),
            (low - out['previous_close']).abs()
        ], axis=1).max(axis=1).where(valid & out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['log_range_squared'] = np.log(high.where(valid) / low.where(valid)).pow(2)
        out['annual_volatility'] = np.sqrt(
            out['log_range_squared'].rolling(params.baseline_window).mean()
            * 252.0 / (4.0 * np.log(2.0))
        )
        out['strength'] = (out['score'].abs() / (3.0 * params.transport_threshold)).clip(0, 1).fillna(0.0)
        ready = out[['score', 'expansion', 'atr', 'annual_volatility']].notna().all(axis=1)
        eligible = ready & (out['atr'] > 0) & (out['annual_volatility'] > 0)
        eligible &= out['score'].abs().ge(params.transport_threshold)
        eligible &= out['expansion'].ge(params.expansion_threshold)
        out['eligible_direction'] = np.sign(out['score'].fillna(0)).where(eligible, 0).astype(int)
        out['entry_direction'] = out['eligible_direction'].where(
            out['eligible_direction'].ne(out['eligible_direction'].shift(1).fillna(0)), 0
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeTransportParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.01, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        volatility = indicators['annual_volatility'].to_numpy(dtype=float)
        strength = indicators['strength'].to_numpy(dtype=float)
        entries = indicators['entry_direction'].to_numpy(dtype=int)
        side = 0
        entry = np.nan
        best = 0.0
        stop = 0.0
        age = 0
        activated = False
        for i in range(n):
            if np.isfinite(volatility[i]) and volatility[i] > 0 and np.isfinite(strength[i]):
                sizes[i] = float(np.clip(
                    params.target_volatility / volatility[i] * strength[i], 0.01, 0.95
                ))
            exited = False
            if side:
                if np.isnan(entry):
                    # The preceding close decision fills at this bar's open.
                    if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                        side = 0
                        exited = True
                    else:
                        entry = opens[i]
                        best = side * entry
                        stop = best - params.trail_atr * atr[i - 1]
                        activated = False
                        age = 0
                if side:
                    age += 1
                    if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                        side = 0
                        exited = True
                    else:
                        signed_close = side * closes[i]
                        best = max(best, signed_close)
                        if (best - side * entry) / entry >= params.breakeven_pct:
                            activated = True
                        if activated:
                            # Signed-price stops only rise; a short's actual stop only falls.
                            stop = max(stop, side * entry, best - params.trail_atr * atr[i])
                        # ponytail: close-confirmed exits; intrabar protection requires an execution stop order.
                        if signed_close <= stop or age >= params.holding_bars:
                            side = 0
                            exited = True
                if exited:
                    entry = np.nan
            if not side and not exited and entries[i] != 0:
                if np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                    side = int(entries[i])
                    entry = np.nan
            signals[i] = side
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7)
    close = np.array([100., 103., 100., 100., 97., 100., 100.])
    opening = np.array([100., 101., 103., 100., 100., 97., 100.])
    data = pd.DataFrame({
        'open': opening, 'high': np.maximum(opening, close) + 1,
        'low': np.minimum(opening, close) - 1, 'close': close,
        'volume': np.full(7, 1000.)
    }, index=index)
    ind = pd.DataFrame({
        'atr': np.ones(7), 'annual_volatility': np.full(7, 0.2),
        'strength': np.full(7, 0.5),
        'entry_direction': [1, 0, 0, -1, 0, 0, 0]
    }, index=index)
    strategy = GeneratedStrategy()
    params = RangeTransportParams()
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert np.allclose(result['size'].iloc[1:], 0.3)
    assert strategy.warmup_bars(params) >= 31
    warmup = strategy.indicators(data, params)
    assert warmup['entry_direction'].eq(0).all()
    params.holding_bars = 2
    flat = data.copy()
    flat['open'] = flat['close'] = 100.0
    expiry = strategy.generate_signals(flat, ind, None, params).data
    assert expiry['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]


if __name__ == '__main__':
    _self_check()
