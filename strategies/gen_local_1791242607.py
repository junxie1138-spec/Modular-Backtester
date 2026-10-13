from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 16
    bridge_threshold: float = 0.60
    atr_window: int = 14
    vol_window: int = 20
    target_vol: float = 0.12
    stop_atr: float = 1.50

    def __post_init__(self):
        if self.graph_window < 6 or self.atr_window < 1 or self.vol_window < 2:
            raise ValueError('Invalid lookback windows')
        if not 0.0 < self.bridge_threshold <= 1.0:
            raise ValueError('bridge_threshold must be in (0, 1]')
        if not np.isfinite(self.target_vol) or self.target_vol <= 0.0:
            raise ValueError('target_vol must be finite and positive')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0.0:
            raise ValueError('stop_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local_1791242607'

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return max(params.graph_window + 1, params.atr_window + 1, params.vol_window + 1)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0.0)
        out['log_close'] = np.log(close)
        out['return'] = close.pct_change(fill_method=None)
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['annual_vol'] = out['return'].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)

        # Natural visibility: a distant slope must be below every nearer slope.
        nearest_min = pd.Series(np.inf, index=data.index, dtype=float)
        bridges = pd.Series(0.0, index=data.index)
        minimum_span = max(2, params.graph_window // 3)
        for distance in range(1, params.graph_window):
            slope = (out['log_close'] - out['log_close'].shift(distance)) / distance
            visible = slope.lt(nearest_min)
            if distance >= minimum_span:
                bridges += (visible & slope.gt(0.0)).astype(float)
            nearest_min = pd.Series(
                np.minimum(nearest_min.to_numpy(), slope.to_numpy()),
                index=data.index,
            )
        valid_graph = out['log_close'].rolling(params.graph_window, min_periods=params.graph_window).count().eq(params.graph_window)
        out['upward_bridge_density'] = (bridges / (params.graph_window - minimum_span)).where(valid_graph)
        out['eligible'] = (
            out['upward_bridge_density'].ge(params.bridge_threshold)
            & out['atr'].gt(0.0)
            & out['annual_vol'].gt(0.0)
            & np.isfinite(out['atr'])
            & np.isfinite(out['annual_vol'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        eligible = indicators['eligible'].fillna(False).astype(bool)
        confirmed = (eligible & eligible.shift(1, fill_value=False)).to_numpy()
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        vol = indicators['annual_vol'].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.95, dtype=float)
        active = False
        entry_bar = -1
        entry_atr = 0.0
        stop = np.nan
        held_size = 0.95

        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # The previous close scheduled this entry for today's open.
                    stop = opening[i] - params.stop_atr * entry_atr
                invalid = not np.isfinite(close[i]) or not np.isfinite(stop)
                stopped = close[i] < stop
                expired = i - entry_bar + 1 >= 2
                if invalid or stopped or expired:
                    active = False
                else:
                    signal[i] = 1
                    size[i] = held_size
                # An exit cannot simultaneously schedule another entry.
                continue

            if confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0.0 and np.isfinite(vol[i]) and vol[i] > 0.0:
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
                held_size = float(np.clip(params.target_vol / vol[i], 0.05, 0.95))
                signal[i] = 1
                size[i] = held_size

        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = VisibilityParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 97.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        'eligible': [False, True, True, False, False, False],
        'atr': 1.0, 'annual_vol': 0.2,
    }, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 0]
    assert np.isclose(result['size'].iloc[3], 0.6)
    data.loc[index[3], 'close'] = 98.0
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 0, 0]
    data.loc[index[3], ['open', 'close']] = [90.0, 89.5]
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 0]
    assert result['size'].gt(0.0).all()
