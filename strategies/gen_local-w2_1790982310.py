from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolumeEchoParams:
    volume_window: int = 20
    dependence_window: int = 60
    min_correlation: float = 0.15
    min_volume_ratio: float = 1.20
    min_move_sigma: float = 0.50
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[VolumeEchoParams]):
    strategy_id = 'gen_local-w2_1790982310'

    @classmethod
    def params_type(cls):
        return VolumeEchoParams

    def warmup_bars(self, params: VolumeEchoParams) -> int:
        return int(params.volume_window + params.dependence_window + 2)

    def indicators(self, data: pd.DataFrame, params: VolumeEchoParams) -> pd.DataFrame:
        if params.volume_window < 2 or params.dependence_window < 3:
            raise ValueError('Lookback windows must be at least 2 and 3 bars.')
        if not 0 < params.min_correlation < 1:
            raise ValueError('min_correlation must be between zero and one.')
        if params.min_volume_ratio <= 0 or params.min_move_sigma <= 0:
            raise ValueError('Entry thresholds must be positive.')
        if params.profit_target <= 0 or params.max_hold_bars < 1:
            raise ValueError('Exit parameters must be positive.')

        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] >= 0)
        out['return'] = close.pct_change(fill_method=None)
        out['volume_mean'] = volume.rolling(params.volume_window).mean().shift(1)
        out['volume_ratio'] = volume.div(out['volume_mean'].replace(0, np.nan))
        out['impulse'] = out['return'] * out['volume_ratio'].clip(0, 3)
        out['lag_impulse'] = out['impulse'].shift(1)
        out['correlation'] = out['return'].rolling(params.dependence_window).corr(out['lag_impulse']).shift(1)
        out['volatility'] = out['return'].rolling(params.dependence_window).std().shift(1)
        out['move_sigma'] = out['return'].div(out['volatility'].replace(0, np.nan))
        out['strength'] = (out['correlation'].clip(0, 1) * out['move_sigma'].clip(0, 3) * out['volume_ratio'].clip(0, 3)).clip(0, 1)
        out['entry'] = (
            (out['correlation'] >= params.min_correlation)
            & (out['volume_ratio'] >= params.min_volume_ratio)
            & (out['move_sigma'] >= params.min_move_sigma)
        )
        numeric = out.columns.drop('entry')
        out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out['entry'] = out['entry'].fillna(False).astype(bool)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VolumeEchoParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strengths = indicators['strength'].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1).to_numpy(dtype=float)
        active = False
        entry_price = np.nan
        entry_bar = -1
        held_size = 1.0

        for i in range(len(data)):
            if active:
                # A preceding close decision becomes an entry at this bar's open.
                if np.isnan(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_up:
                    active = False
                    entry_price = np.nan
                else:
                    signals[i] = 1
                    sizes[i] = held_size
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_price = np.nan
                held_size = float(0.25 + 0.70 * strengths[i])
                signals[i] = 1
                sizes[i] = held_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    data.loc[index[2], 'close'] = 103.0
    indicators = pd.DataFrame({'entry': False, 'strength': 0.5}, index=index)
    indicators.loc[index[[0, 4]], 'entry'] = True
    params = VolumeEchoParams(profit_target=0.02, max_hold_bars=3)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    assert np.isclose(result.loc[index[1], 'size'], 0.60)
    assert (result['size'] > 0).all()
    warmup = strategy.indicators(data, params)
    assert not warmup['entry'].any()
    assert np.isfinite(warmup.drop(columns='entry').to_numpy()).all()


if __name__ == '__main__':
    _self_check()
