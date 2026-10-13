from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    fast_window: int = 6
    slow_window: int = 40
    atr_window: int = 20
    compression_ratio: float = 0.65
    pressure_atr: float = 1.0
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = 'gen_local-w2_1791334775'

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    @staticmethod
    def warmup_bars(params: CompressionParams) -> int:
        return int(max(params.fast_window, params.slow_window, params.atr_window) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if not (2 <= params.fast_window < params.slow_window and params.atr_window >= 2):
            raise ValueError('Require 2 <= fast_window < slow_window and atr_window >= 2')
        if not (0 < params.compression_ratio < 1 and params.pressure_atr > 0):
            raise ValueError('Require a compression ratio in (0, 1) and positive pressure_atr')
        if not (params.profit_target > 0 and params.max_hold_bars >= 1):
            raise ValueError('Require positive profit_target and max_hold_bars')
        close = data['close'].where(data['close'] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        fast_vol = returns.rolling(params.fast_window, min_periods=params.fast_window).std(ddof=0)
        slow_vol = returns.rolling(params.slow_window, min_periods=params.slow_window).std(ddof=0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        volatility_ratio = fast_vol.div(slow_vol.where(slow_vol > 0))
        compressed = (volatility_ratio <= params.compression_ratio).fillna(False)
        episode_start = compressed & ~compressed.shift(1, fill_value=False)
        anchor = previous_close.where(episode_start).ffill().where(compressed)
        frozen_atr = atr.shift(1).where(episode_start).ffill().where(compressed)
        pressure = (close - anchor).div(frozen_atr.where(frozen_atr > 0))
        eligible = (compressed & (pressure >= params.pressure_atr)).fillna(False)
        episode = episode_start.cumsum().astype(int)
        confirmed = eligible & eligible.shift(1, fill_value=False)
        confirmed &= episode.eq(episode.shift(1))
        return pd.DataFrame({
            'returns': returns,
            'fast_vol': fast_vol,
            'slow_vol': slow_vol,
            'true_range': true_range,
            'atr': atr,
            'volatility_ratio': volatility_ratio,
            'compressed': compressed,
            'episode_start': episode_start,
            'episode': episode,
            'anchor': anchor,
            'frozen_atr': frozen_atr,
            'pressure': pressure,
            'eligible': eligible,
            'confirmed': confirmed.fillna(False),
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        episodes = indicators['episode'].to_numpy(dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        traded_episode = -1
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # The prior close's decision fills at this bar's open.
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                        continue
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or expired:
                    active = False
                    continue
                signals[i] = 1
            elif confirmed[i] and episodes[i] != traded_episode:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                traded_episode = episodes[i]
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = CompressionParams(max_hold_bars=3)
    index = pd.date_range('2020-01-01', periods=7, freq='D')
    data = pd.DataFrame({
        'open': [100.0] * 7,
        'high': [101.0] * 7,
        'low': [99.0] * 7,
        'close': [100.0] * 7,
        'volume': [1000.0] * 7,
    }, index=index)
    derived = pd.DataFrame({
        'confirmed': [False, True, True, True, True, True, True],
        'episode': [1] * 7,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, 0]
    data.loc[index[2], 'open'] = 102.0
    data.loc[index[2], 'close'] = 104.5
    data.loc[index[3], 'close'] = 107.0
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0, 0]
    calculated = GeneratedStrategy.indicators(data, CompressionParams())
    assert calculated.index.equals(data.index)
    assert not calculated['confirmed'].any()
    assert GeneratedStrategy.warmup_bars(CompressionParams()) == 41


if __name__ == '__main__':
    _self_check()
