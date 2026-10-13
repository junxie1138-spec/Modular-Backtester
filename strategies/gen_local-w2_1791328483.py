from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeMigrationParams:
    block_bars: int = 5
    channel_bars: int = 40
    atr_bars: int = 14
    novelty_threshold: float = 0.65
    edge_threshold: float = 0.65
    atr_multiple: float = 2.8
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[RangeMigrationParams]):
    strategy_id = 'gen_local-w2_1791328483'

    @classmethod
    def params_type(cls) -> type[RangeMigrationParams]:
        return RangeMigrationParams

    def warmup_bars(self, params: RangeMigrationParams) -> int:
        return int(max(200, params.channel_bars, 2 * params.block_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: RangeMigrationParams) -> pd.DataFrame:
        if min(params.block_bars, params.channel_bars, params.atr_bars, params.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0 < params.novelty_threshold < 1 or not 0.5 < params.edge_threshold < 1:
            raise ValueError('Invalid overlap or channel threshold.')
        if not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError('ATR multiple must be finite and positive.')
        out = pd.DataFrame(index=data.index)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['block_high'] = data['high'].rolling(params.block_bars, min_periods=params.block_bars).max()
        out['block_low'] = data['low'].rolling(params.block_bars, min_periods=params.block_bars).min()
        out['prior_high'] = out['block_high'].shift(params.block_bars)
        out['prior_low'] = out['block_low'].shift(params.block_bars)
        out['intersection'] = (np.minimum(out['block_high'], out['prior_high']) - np.maximum(out['block_low'], out['prior_low'])).clip(lower=0)
        out['union'] = np.maximum(out['block_high'], out['prior_high']) - np.minimum(out['block_low'], out['prior_low'])
        out['novelty'] = 1.0 - out['intersection'] / out['union'].where(out['union'] > 0)
        out['previous_novelty'] = out['novelty'].shift(1)
        out['midpoint_move'] = (out['block_high'] + out['block_low'] - out['prior_high'] - out['prior_low']) / 2.0
        out['channel_high'] = data['high'].rolling(params.channel_bars, min_periods=params.channel_bars).max()
        out['channel_low'] = data['low'].rolling(params.channel_bars, min_periods=params.channel_bars).min()
        out['channel_width'] = out['channel_high'] - out['channel_low']
        out['position'] = (data['close'] - out['channel_low']) / out['channel_width'].where(out['channel_width'] > 0)
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeMigrationParams) -> SignalFrame:
        required = ['ma200', 'novelty', 'previous_novelty', 'midpoint_move', 'position', 'atr']
        valid = np.isfinite(indicators[required]).all(axis=1) & np.isfinite(data['close']) & (indicators['atr'] > 0)
        trigger = valid & (indicators['novelty'] >= params.novelty_threshold) & (indicators['previous_novelty'] < params.novelty_threshold)
        long_entry = trigger & (indicators['midpoint_move'] > 0) & (indicators['position'] >= params.edge_threshold) & (data['close'] > indicators['ma200'])
        short_entry = trigger & (indicators['midpoint_move'] < 0) & (indicators['position'] <= 1.0 - params.edge_threshold) & (data['close'] < indicators['ma200'])
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        averages = indicators['ma200'].to_numpy(dtype=float)
        usable = valid.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        age = 0
        extreme = np.nan
        stop = np.nan
        for i in range(len(data)):
            if side:
                age += 1
                if not usable[i]:
                    side = 0
                else:
                    close = closes[i]
                    distance = params.atr_multiple * atrs[i]
                    if side == 1:
                        # Highest close since the next-bar entry; the stop only rises.
                        extreme = close if np.isnan(extreme) else max(extreme, close)
                        candidate = extreme - distance
                        stop = candidate if np.isnan(stop) else max(stop, candidate)
                        stopped = close < stop
                        regime_exit = close <= averages[i]
                    else:
                        # Mirror the close-based trailing stop for short positions.
                        extreme = close if np.isnan(extreme) else min(extreme, close)
                        candidate = extreme + distance
                        stop = candidate if np.isnan(stop) else min(stop, candidate)
                        stopped = close > stop
                        regime_exit = close >= averages[i]
                    if stopped or regime_exit or age >= params.hold_bars:
                        side = 0
                signals[i] = side
                # Preserve an exit as a flat decision, even if a fresh trigger fires.
                continue
            if usable[i] and entries[i]:
                side = int(entries[i])
                age = 0
                extreme = np.nan
                stop = np.nan
            signals[i] = side
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = RangeMigrationParams(atr_multiple=2.0)
    index = pd.date_range('2020-01-01', periods=8)
    for side in (1, -1):
        close = np.array([100, 100, 102, 106, 105, 101, 100, 99], dtype=float)
        if side == -1:
            close = 200.0 - close
        data = pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': 1000.0}, index=index)
        indicators = pd.DataFrame({
            'ma200': 90.0 if side == 1 else 110.0,
            'novelty': [0.8] + [0.0] * 7,
            'previous_novelty': 0.0,
            'midpoint_move': float(side),
            'position': 0.9 if side == 1 else 0.1,
            'atr': [1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 1.0, 1.0],
        }, index=index)
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
        assert result['signal'].tolist() == [0, side, side, side, side, side, 0, 0]
        assert (result['size'] > 0).all()
        indicators.loc[:, 'ma200'] = np.nan
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
        assert (result['signal'] == 0).all()


if __name__ == '__main__':
    _self_check()
