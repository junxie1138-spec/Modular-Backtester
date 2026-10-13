from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    segment_bars: int = 2
    volume_bars: int = 20
    atr_bars: int = 14
    analog_bars: int = 252
    percentile_bars: int = 126
    min_analogs: int = 12
    percentile: float = 0.7
    hold_bars: int = 8
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local-w2_1791255206'

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(max(
            params.atr_bars + 1,
            params.volume_bars + 3 * params.segment_bars
            + params.hold_bars + params.analog_bars
            + params.percentile_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        p = params
        windows = (p.segment_bars, p.volume_bars, p.atr_bars,
                   p.analog_bars, p.percentile_bars, p.hold_bars)
        if any(w < 1 for w in windows) or p.min_analogs < 1:
            raise ValueError('Lookbacks, holding period, and minimum analogs must be positive')
        if not 0.0 < p.percentile < 1.0 or not np.isfinite(p.stop_atr) or p.stop_atr <= 0:
            raise ValueError('Invalid percentile or ATR multiplier')

        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] >= 0)
        out['return'] = close.pct_change(fill_method=None)
        out['volume_baseline'] = volume.shift(1).rolling(p.volume_bars).mean()
        out['relative_volume'] = volume / out['volume_baseline'].replace(0.0, np.nan)
        out['weighted_return'] = out['return'] * out['relative_volume']
        out['segment_recent'] = out['weighted_return'].rolling(p.segment_bars).sum()
        out['segment_middle'] = out['segment_recent'].shift(p.segment_bars)
        out['segment_oldest'] = out['segment_recent'].shift(2 * p.segment_bars)
        segments = out[['segment_oldest', 'segment_middle', 'segment_recent']]
        valid_shape = np.isfinite(segments).all(axis=1)
        # Eight ordinal cohorts preserve the sequence of buying and selling pressure.
        out['shape'] = (
            4 * out['segment_oldest'].gt(0).astype(int)
            + 2 * out['segment_middle'].gt(0).astype(int)
            + out['segment_recent'].gt(0).astype(int)
        ).astype(float).where(valid_shape)
        out['shape_move'] = close / close.shift(3 * p.segment_bars) - 1.0
        out['confirmation_volume'] = out['relative_volume'].rolling(p.segment_bars).mean()
        out['volume_threshold'] = out['confirmation_volume'].shift(1).rolling(
            p.percentile_bars
        ).quantile(p.percentile)

        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(p.atr_bars).mean()

        # At t, this is the completed outcome of the shape observed at t-hold_bars.
        out['matured_shape'] = out['shape'].shift(p.hold_bars)
        out['matured_return'] = close / close.shift(p.hold_bars) - 1.0
        out['forecast'] = np.nan
        for code in range(8):
            matched = out['matured_shape'].eq(code) & np.isfinite(out['matured_return'])
            out[f'count_{code}'] = matched.astype(float).rolling(p.analog_bars).sum()
            out[f'outcome_sum_{code}'] = out['matured_return'].where(
                matched, 0.0
            ).rolling(p.analog_bars).sum()
            out[f'mean_{code}'] = (
                out[f'outcome_sum_{code}'] / out[f'count_{code}'].replace(0.0, np.nan)
            ).where(out[f'count_{code}'] >= p.min_analogs)
            out['forecast'] = out['forecast'].where(
                ~out['shape'].eq(code), out[f'mean_{code}']
            )

        out['forecast_threshold'] = out['forecast'].shift(1).rolling(
            p.percentile_bars, min_periods=p.min_analogs
        ).quantile(p.percentile)
        required = ['forecast', 'forecast_threshold', 'confirmation_volume',
                    'volume_threshold', 'atr', 'shape_move', 'segment_recent']
        ready = np.isfinite(out[required]).all(axis=1)
        out['entry'] = (
            ready
            & out['forecast'].gt(out['forecast_threshold'])
            & out['forecast'].gt(0)
            & out['confirmation_volume'].gt(out['volume_threshold'])
            & out['segment_recent'].gt(0)
            & out['shape_move'].gt(0)
            & out['atr'].gt(0)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShapeParams,
    ) -> SignalFrame:
        n = len(data)
        decisions = np.zeros(n, dtype=int)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        active = False
        stop = np.nan
        held = 0

        for i in range(n):
            # A decision made yesterday enters at today's open.
            if i > 0 and decisions[i - 1] == 1 and not active:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]):
                    active = True
                    held = 0
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                else:
                    continue
            if active:
                held += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or held >= params.hold_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                decisions[i] = 1

        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 96.0,
        'close': [100.0, 100.0, 97.0, 99.0, 99.0], 'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'entry': [True, False, False, False, False], 'atr': 1.0,
    }, index=index)
    strategy = GeneratedStrategy()
    stopped = strategy.generate_signals(data, indicators, None, ShapeParams())
    assert stopped.data['signal'].tolist() == [0, 1, 1, 0, 0]
    data['close'] = 100.0
    expired = strategy.generate_signals(data, indicators, None, ShapeParams(hold_bars=2))
    assert expired.data['signal'].tolist() == [0, 1, 1, 0, 0]
    assert (expired.data['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
