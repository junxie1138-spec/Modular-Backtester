from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_bars: int = 12
    squeeze_bars: int = 8
    baseline_bars: int = 40
    compression_ratio: float = 0.80
    topology_threshold: float = 0.35
    arm_profit_pct: float = 0.02
    initial_stop_atr: float = 2.0
    trail_atr: float = 2.5
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local_1791195892'

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(max(params.graph_bars + 1, params.squeeze_bars + 1, params.baseline_bars + 1) + 8)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if min(params.graph_bars, params.squeeze_bars, params.baseline_bars) < 2:
            raise ValueError('Indicator windows must be at least two bars.')
        if not 0 < params.compression_ratio < 1 or not 0 < params.topology_threshold < 1:
            raise ValueError('Compression and topology thresholds must lie between zero and one.')
        if min(params.arm_profit_pct, params.initial_stop_atr, params.trail_atr) <= 0 or params.max_hold_bars < 1:
            raise ValueError('Exit parameters must be positive.')

        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.squeeze_bars, min_periods=params.squeeze_bars).mean()
        out['baseline_atr'] = out['true_range'].rolling(params.baseline_bars, min_periods=params.baseline_bars).mean()
        out['range_span'] = data['high'].rolling(params.squeeze_bars, min_periods=params.squeeze_bars).max() - data['low'].rolling(params.squeeze_bars, min_periods=params.squeeze_bars).min()
        out['compression_ratio'] = out['atr'] / out['baseline_atr'].replace(0.0, np.nan)
        out['compressed'] = (out['compression_ratio'] < params.compression_ratio) & (out['range_span'] <= 4.0 * out['baseline_atr'])
        out['recent_compression'] = out['compressed'].astype(float).rolling(6, min_periods=6).max().shift(1).eq(1.0)

        upper_block = pd.Series(-np.inf, index=data.index)
        lower_block = pd.Series(np.inf, index=data.index)
        upper_degree = pd.Series(0.0, index=data.index)
        lower_degree = pd.Series(0.0, index=data.index)
        upper_reach = pd.Series(0.0, index=data.index)
        lower_reach = pd.Series(0.0, index=data.index)
        # An edge exists only when every intervening close lies below both endpoints.
        # Reflecting prices gives the corresponding downward visibility graph.
        for lag in range(1, params.graph_bars + 1):
            past = close.shift(lag)
            finite = np.isfinite(close) & np.isfinite(past)
            upper_edge = finite & (np.minimum(close, past) > upper_block)
            lower_edge = finite & (np.maximum(close, past) < lower_block)
            upper_degree += upper_edge.astype(float)
            lower_degree += lower_edge.astype(float)
            upper_reach += lag * upper_edge.astype(float)
            lower_reach += lag * lower_edge.astype(float)
            upper_block = np.maximum(upper_block, past)
            lower_block = np.minimum(lower_block, past)

        complete_graph = close.where(np.isfinite(close)).rolling(params.graph_bars + 1, min_periods=params.graph_bars + 1).count().eq(params.graph_bars + 1)
        out['upper_degree'] = upper_degree.where(complete_graph)
        out['lower_degree'] = lower_degree.where(complete_graph)
        out['upper_reach'] = upper_reach.where(complete_graph)
        out['lower_reach'] = lower_reach.where(complete_graph)
        out['topology_score'] = (out['upper_reach'] - out['lower_reach']) / (out['upper_reach'] + out['lower_reach']).replace(0.0, np.nan)
        valid_prices = np.isfinite(data[['open', 'high', 'low', 'close']]).all(axis=1) & (close > 0)
        ready = valid_prices & complete_graph & np.isfinite(out['atr']) & (out['atr'] > 0) & np.isfinite(out['baseline_atr']) & out['recent_compression']
        long_setup = ready & (out['topology_score'] >= params.topology_threshold) & (out['upper_degree'] >= 3)
        short_setup = ready & (out['topology_score'] <= -params.topology_threshold) & (out['lower_degree'] >= 3)
        out['candidate'] = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        out['confirmed'] = out['candidate'].where(out['candidate'].eq(out['candidate'].shift(1)), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        highs = data['high'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry = np.nan
        stop = 0.0
        peak = 0.0
        age = 0
        armed = False

        for i in range(len(data)):
            valid = np.isfinite([opens[i], highs[i], lows[i], closes[i], atrs[i]]).all() and min(opens[i], closes[i], atrs[i]) > 0
            if side:
                if not valid:
                    side = 0
                    continue
                # A decision from the previous close becomes exposure at this open.
                if np.isnan(entry):
                    entry = opens[i]
                age += 1
                # Store the stop in signed profit coordinates: it only moves upward.
                # This raises long price stops and lowers short price stops.
                adverse = lows[i] - entry if side == 1 else entry - highs[i]
                if adverse <= stop or age >= params.max_hold_bars:
                    side = 0
                else:
                    profit = side * (closes[i] - entry)
                    peak = max(peak, profit)
                    armed = armed or peak >= entry * params.arm_profit_pct
                    if armed:
                        stop = max(stop, 0.0, peak - params.trail_atr * atrs[i])
                    if profit <= stop:
                        side = 0
                # Stop breaches request a flat position at the next open.
                decisions[i] = side
                continue

            if valid and confirmed[i] in (-1, 1):
                side = int(confirmed[i])
                entry = np.nan
                stop = -params.initial_stop_atr * atrs[i]
                peak = 0.0
                age = 0
                armed = False
                decisions[i] = side

        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = VisibilityParams()
    index = pd.date_range('2020-01-01', periods=8)
    for direction in (1, -1):
        data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
        data.loc[index[2], 'close'] = 100.0 + direction * 3.0
        data.loc[index[2], 'high'] = 104.0 if direction == 1 else 101.0
        data.loc[index[2], 'low'] = 96.0 if direction == -1 else 99.0
        indicators = pd.DataFrame({'atr': 1.0, 'confirmed': [0, direction, 0, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, None, params)
        assert result.data['signal'].tolist() == [0, 0, direction, direction, 0, 0, 0, 0]
        assert result.data['size'].gt(0).all()
