from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletAnalogParams:
    energy_window: int = 8
    migration_bars: int = 5
    history_bars: int = 252
    rank_window: int = 126
    neighbor_percentile: float = 0.10
    entry_percentile: float = 0.75
    profit_target: float = 0.05
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletAnalogParams]):
    strategy_id = 'gen_local-w2_1791250694'

    @classmethod
    def params_type(cls):
        return WaveletAnalogParams

    def warmup_bars(self, params: WaveletAnalogParams) -> int:
        return int(16 + params.energy_window + params.migration_bars
                   + params.history_bars + params.hold_bars
                   + params.rank_window + 1)

    def indicators(self, data: pd.DataFrame, params: WaveletAnalogParams) -> pd.DataFrame:
        if min(params.energy_window, params.migration_bars,
               params.history_bars, params.rank_window, params.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0 < params.neighbor_percentile <= 1:
            raise ValueError('neighbor_percentile must be in (0, 1].')
        if not 0 < params.entry_percentile < 1 or params.profit_target <= 0:
            raise ValueError('Invalid entry percentile or profit target.')

        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['log_return'] = np.log(close).diff()
        energy_columns = []
        for scale in (2, 4, 8):
            mean_column = f'mean_{scale}'
            haar_column = f'haar_{scale}'
            energy_column = f'energy_{scale}'
            out[mean_column] = out['log_return'].rolling(scale).mean()
            out[haar_column] = np.sqrt(scale / 2.0) * (
                out[mean_column] - out[mean_column].shift(scale)
            )
            out[energy_column] = out[haar_column].pow(2).rolling(
                params.energy_window
            ).mean()
            energy_columns.append(energy_column)

        out['total_energy'] = out[energy_columns].sum(axis=1, min_count=3)
        denominator = out['total_energy'].where(out['total_energy'] > 0)
        feature_columns = []
        for scale in (2, 4, 8):
            share_column = f'share_{scale}'
            migration_column = f'migration_{scale}'
            out[share_column] = out[f'energy_{scale}'] / denominator
            out[migration_column] = out[share_column].diff(params.migration_bars)
            feature_columns.extend((share_column, migration_column))

        features = out[feature_columns].to_numpy(dtype=float)
        closes = close.to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        analog_count = np.zeros(len(data), dtype=int)
        horizon = params.hold_bars
        first_feature = 16 + params.energy_window + params.migration_bars

        # ponytail: O(bars * history), batch distances if histories become large.
        for i in range(first_feature + params.history_bars + horizon, len(data)):
            if not np.isfinite(features[i]).all():
                continue
            # Every outcome ends before the current decision bar.
            candidates = np.arange(i - horizon - params.history_bars, i - horizon)
            valid = np.isfinite(features[candidates]).all(axis=1)
            valid &= np.isfinite(opens[candidates + 1]) & (opens[candidates + 1] > 0)
            valid &= np.isfinite(closes[candidates + horizon])
            candidates = candidates[valid]
            if len(candidates) < 20:
                continue
            distance = np.square(features[candidates] - features[i]).sum(axis=1)
            cutoff = np.quantile(distance, params.neighbor_percentile)
            neighbors = candidates[distance <= cutoff]
            outcomes = closes[neighbors + horizon] / opens[neighbors + 1] - 1.0
            forecast[i] = float(outcomes.mean())
            analog_count[i] = len(neighbors)

        out['analog_count'] = analog_count
        out['forecast'] = forecast
        out['entry_threshold'] = out['forecast'].shift(1).rolling(
            params.rank_window, min_periods=params.rank_window
        ).quantile(params.entry_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletAnalogParams,
    ) -> SignalFrame:
        eligible = (
            indicators['forecast'].gt(indicators['entry_threshold'])
            & indicators['forecast'].gt(0.0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    active = False
                    continue
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_stop = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_stop:
                    active = False
                else:
                    signals[i] = 1
            elif eligible[i] and i + 1 < len(data):
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                signals[i] = 1

        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({
        'open': 100.0,
        'high': 107.0,
        'low': 99.0,
        'close': [100.0, 100.0, 106.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'forecast': [1.0, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, np.nan],
        'entry_threshold': 0.0,
    }, index=index)
    params = WaveletAnalogParams(hold_bars=3)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1, 1, 0]
    assert result['size'].gt(0).all()
    assert strategy.indicators(data, params)['forecast'].isna().all()


if __name__ == '__main__':
    _self_check()
