from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 5
    extreme_window: int = 20
    volume_window: int = 20
    atr_window: int = 14
    excursion: float = 0.01
    area_threshold: float = 0.12
    stop_atr: float = 1.5
    holding_bars: int = 2
    min_size: float = 0.25
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790932614"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return max(2, params.signature_window, params.extreme_window,
                   params.volume_window, params.atr_window) + 1

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        window = max(2, params.signature_window)
        extreme = max(2, params.extreme_window)
        volume_window = max(2, params.volume_window)
        atr_window = max(2, params.atr_window)
        close = data['close'].where(data['close'] > 0)
        out['log_price'] = np.log(close)
        out['return'] = out['log_price'].diff()
        out['range'] = (data['high'] - data['low']).clip(lower=0)
        out['volume_mean'] = data['volume'].rolling(volume_window).mean()
        out['relative_volume'] = data['volume'].div(out['volume_mean'].replace(0, np.nan))
        out['closing_pressure'] = ((2 * close - data['high'] - data['low'])
                                   .div(out['range'].replace(0, np.nan))
                                   .clip(-1, 1).fillna(0))
        out['pressure'] = (out['closing_pressure'] * out['relative_volume'])
        out['flow_increment'] = out['pressure'].replace([np.inf, -np.inf], np.nan).fillna(0)
        out['cumulative_flow'] = out['flow_increment'].cumsum()
        out['price_increment'] = out['return'].fillna(0)
        out['cross_increment'] = (
            out['log_price'].shift(1) * out['flow_increment']
            - out['cumulative_flow'].shift(1) * out['price_increment']
        )
        out['price_start'] = out['log_price'].shift(window)
        out['flow_start'] = out['cumulative_flow'].shift(window)
        out['price_change'] = out['log_price'] - out['price_start']
        out['flow_change'] = out['cumulative_flow'] - out['flow_start']
        out['signature_area'] = 0.5 * (
            out['cross_increment'].rolling(window).sum()
            - out['price_start'] * out['flow_change']
            + out['flow_start'] * out['price_change']
        )
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            out['range'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(atr_window).mean()
        out['atr_fraction'] = out['atr'].div(close)
        out['flow_capacity'] = out['flow_increment'].abs().rolling(window).sum()
        out['normalized_area'] = out['signature_area'].div(
            (out['atr_fraction'] * out['flow_capacity']).replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out['peak'] = close.rolling(extreme).max()
        out['trough'] = close.rolling(extreme).min()
        out['drawdown'] = 1 - close.div(out['peak'])
        out['runup'] = close.div(out['trough']) - 1
        threshold = max(0.001, abs(params.area_threshold))
        excursion = max(0.0, params.excursion)
        out['ready'] = (
            out['normalized_area'].notna()
            & out['pressure'].notna()
            & out['drawdown'].notna()
            & out['runup'].notna()
            & out['atr'].gt(0)
        )
        out['long_setup'] = (
            out['ready'] & out['drawdown'].ge(excursion)
            & out['normalized_area'].le(-threshold)
            & out['return'].gt(0) & out['pressure'].gt(0)
        ).fillna(False)
        out['short_setup'] = (
            out['ready'] & out['runup'].ge(excursion)
            & out['normalized_area'].ge(threshold)
            & out['return'].lt(0) & out['pressure'].lt(0)
        ).fillna(False)
        out['strength'] = (out['normalized_area'].abs() / (2 * threshold)).clip(0, 1).fillna(0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        minimum = min(1.0, max(0.01, params.min_size))
        maximum = min(1.0, max(minimum, params.max_size))
        sizes = np.full(count, minimum, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        longs = indicators['long_setup'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_setup'].fillna(False).to_numpy(dtype=bool)
        strength = indicators['strength'].fillna(0).clip(0, 1).to_numpy(dtype=float)
        horizon = min(2, max(1, params.holding_bars))
        stop_multiple = max(0.01, params.stop_atr)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        entry_atr = np.nan
        trade_size = minimum
        for i in range(count):
            if position:
                # The previous close's decision fills at this bar's open.
                if i == entry_bar + 1:
                    entry_price = opens[i]
                age = i - entry_bar
                invalid = not (np.isfinite(entry_price) and entry_price > 0
                               and np.isfinite(closes[i]))
                stopped = (
                    position == 1 and closes[i] < entry_price - stop_multiple * entry_atr
                ) or (
                    position == -1 and closes[i] > entry_price + stop_multiple * entry_atr
                )
                if invalid or stopped or age >= horizon:
                    position = 0
                sizes[i] = trade_size
            elif np.isfinite(atr[i]) and atr[i] > 0:
                direction = 1 if longs[i] else (-1 if shorts[i] else 0)
                if direction:
                    position = direction
                    entry_bar = i
                    entry_price = np.nan
                    entry_atr = atr[i]
                    trade_size = minimum + (maximum - minimum) * strength[i]
                    sizes[i] = trade_size
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(minimum).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7, freq='D')
    data = pd.DataFrame({
        'open': [100.0] * 7,
        'high': [103.0] * 7,
        'low': [97.0] * 7,
        'close': [100.0, 98.0, 100.0, 100.0, 102.0, 100.0, 100.0],
        'volume': [1000.0] * 7,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0] * 7,
        'long_setup': [True, False, False, False, False, False, False],
        'short_setup': [False, False, False, True, False, False, False],
        'strength': [0.5] * 7,
    }, index=index)
    strategy = GeneratedStrategy()
    params = SignatureParams()
    stopped = strategy.generate_signals(data, indicators, None, params).data
    assert stopped['signal'].tolist() == [0, 1, 0, 0, -1, 0, 0]
    assert np.isclose(stopped['size'].iloc[1], 0.625)
    data['close'] = 100.0
    expired = strategy.generate_signals(data, indicators, None, params).data
    assert expired['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert expired['size'].gt(0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived['long_setup'].any()
    assert not derived['short_setup'].any()
