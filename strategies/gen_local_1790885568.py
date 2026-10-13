from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 12
    compression_limit: float = 0.95
    bridge_ratio: float = 1.5
    stop_atr: float = 2.0
    hold_days: int = 4

    def __post_init__(self):
        if self.graph_window < 4:
            raise ValueError('graph_window must be at least 4')
        if not 3 <= self.hold_days <= 5:
            raise ValueError('hold_days must be between 3 and 5')
        if not all(np.isfinite(v) and v > 0 for v in (
            self.compression_limit, self.bridge_ratio, self.stop_atr
        )):
            raise ValueError('thresholds must be finite and positive')


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local_1790885568'

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return max(200, params.graph_window + 1, 61)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        close = data['close'].astype(float)
        previous = close.shift(1)
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out['atr'] = tr.rolling(14, min_periods=14).mean()
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        w = params.graph_window
        out['box_high'] = data['high'].shift(1).rolling(w, min_periods=w).max()
        out['box_low'] = data['low'].shift(1).rolling(w, min_periods=w).min()
        out['baseline_atr'] = tr.shift(1).rolling(60, min_periods=60).mean()
        out['compression'] = (out['box_high'] - out['box_low']) / (
            out['baseline_atr'].where(out['baseline_atr'] > 0) * np.sqrt(w)
        )
        path = np.column_stack([close.shift(j).to_numpy() for j in range(w, 0, -1)])
        valid = np.isfinite(path).all(axis=1)
        upper = np.zeros(len(data), dtype=float)
        lower = np.zeros(len(data), dtype=float)
        cut = w // 2
        # Exact natural-visibility edges crossing the temporal midpoint.
        # ponytail: cubic window work; vectorize intermediate vertices if windows grow large.
        for a in range(cut):
            for b in range(cut, w):
                visible_upper = valid.copy()
                visible_lower = valid.copy()
                for k in range(a + 1, b):
                    chord = path[:, a] + (path[:, b] - path[:, a]) * ((k - a) / (b - a))
                    visible_upper &= path[:, k] < chord
                    visible_lower &= path[:, k] > chord
                upper += visible_upper
                lower += visible_lower
        out['upper_bridges'] = np.where(valid, upper, np.nan)
        out['lower_bridges'] = np.where(valid, lower, np.nan)
        ready = np.isfinite(out.to_numpy()).all(axis=1) & (out['atr'] > 0)
        compressed = ready & (out['compression'] <= params.compression_limit)
        out['long_entry'] = (
            compressed & (close > out['ma200']) & (close > out['box_high'])
            & (out['upper_bridges'] >= params.bridge_ratio * out['lower_bridges'])
        ).fillna(False)
        out['short_entry'] = (
            compressed & (close < out['ma200']) & (close < out['box_low'])
            & (out['lower_bridges'] >= params.bridge_ratio * out['upper_bridges'])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: VisibilityParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        longs = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        position = 0
        age = 0
        high_water = low_water = 0.0
        rising_stop = -np.inf
        falling_stop = np.inf
        for i in range(len(data)):
            c, atr = closes[i], atrs[i]
            valid = np.isfinite(c) and np.isfinite(atr) and c > 0 and atr > 0
            if position:
                if age == 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    high_water = low_water = opens[i]
                    rising_stop = -np.inf
                    falling_stop = np.inf
                age += 1
                if not valid:
                    position = 0
                    continue
                high_water = max(high_water, c)
                low_water = min(low_water, c)
                rising_stop = max(rising_stop, high_water - params.stop_atr * atr)
                falling_stop = min(falling_stop, low_water + params.stop_atr * atr)
                # The required rolling-high stop applies to both directions.
                # Shorts also receive a mirrored stop against adverse upward moves.
                exit_now = (
                    c <= rising_stop
                    or (position == -1 and c >= falling_stop)
                    or age >= params.hold_days
                )
                if exit_now:
                    position = 0
            elif valid:
                if longs[i]:
                    position = 1
                    age = 0
                elif shorts[i]:
                    position = -1
                    age = 0
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=210, freq='D')
    data = pd.DataFrame({
        'open': 100.0, 'high': 105.0, 'low': 99.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    data.loc[index[201:204], 'close'] = [103.0, 104.0, 100.0]
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    indicators = strategy.indicators(data, params)
    indicators['atr'] = 1.0
    indicators['long_entry'] = False
    indicators['short_entry'] = False
    indicators.loc[index[200], 'long_entry'] = True
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[200:205].tolist() == [0, 1, 1, 1, 0]
    assert result['signal'].isin([-1, 0, 1]).all()
    assert result['size'].gt(0).all()
