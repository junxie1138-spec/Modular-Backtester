from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityRecoveryParams:
    graph_window: int = 24
    min_depth: float = 0.025
    min_bridges: int = 2
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[VisibilityRecoveryParams]):
    strategy_id = 'gen_local_1791221453'

    @classmethod
    def params_type(cls) -> type[VisibilityRecoveryParams]:
        return VisibilityRecoveryParams

    @staticmethod
    def warmup_bars(params: VisibilityRecoveryParams) -> int:
        return max(200, params.graph_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityRecoveryParams) -> pd.DataFrame:
        if params.graph_window < 8 or params.min_bridges < 1:
            raise ValueError('graph_window must be >= 8 and min_bridges >= 1')
        if not np.isfinite(params.min_depth) or params.min_depth <= 0:
            raise ValueError('min_depth must be finite and positive')
        close = data['close'].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0))
        window = params.graph_window
        out = pd.DataFrame(index=data.index)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['peak'] = close.rolling(window, min_periods=window).max()
        out['trough'] = close.rolling(window, min_periods=window).min()
        out['depth'] = out['peak'] / out['trough'] - 1.0
        span = (out['peak'] - out['trough']).replace(0.0, np.nan)
        out['recovery'] = (close - out['trough']) / span
        log_close = np.log(close)
        history = np.column_stack([log_close.shift(k).to_numpy() for k in range(window)])
        valid = np.isfinite(history).all(axis=1)
        trough_age = np.argmin(np.where(np.isfinite(history), history, np.inf), axis=1)
        peak_age = np.argmax(np.where(np.isfinite(history), history, -np.inf), axis=1)
        upper_frontier = np.full(len(data), -np.inf)
        lower_frontier = np.full(len(data), -np.inf)
        upper_bridges = np.zeros(len(data), dtype=int)
        lower_bridges = np.zeros(len(data), dtype=int)
        # An endpoint sees lag k when its slope exceeds all nearer slopes.
        for lag in range(1, window):
            slope = (history[:, lag] - history[:, 0]) / lag
            upper_edge = slope > upper_frontier
            lower_edge = -slope > lower_frontier
            upper_bridges += upper_edge & (lag > trough_age)
            lower_bridges += lower_edge & (lag > peak_age)
            upper_frontier = np.maximum(upper_frontier, slope)
            lower_frontier = np.maximum(lower_frontier, -slope)
        out['trough_age'] = np.where(valid, trough_age, np.nan)
        out['peak_age'] = np.where(valid, peak_age, np.nan)
        out['upper_bridges'] = np.where(valid, upper_bridges, np.nan)
        out['lower_bridges'] = np.where(valid, lower_bridges, np.nan)
        out['new_upper_bridges'] = out['upper_bridges'].diff()
        out['new_lower_bridges'] = out['lower_bridges'].diff()
        out['ready'] = np.isfinite(out.to_numpy(dtype=float)).all(axis=1)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityRecoveryParams) -> SignalFrame:
        if params.max_hold < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError('max_hold must be >= 1 and profit_target finite and positive')
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        ind = indicators
        ready = ind['ready'].fillna(False).astype(bool)
        stressed = ind['depth'].ge(params.min_depth)
        long_entry = (
            ready & stressed & data['close'].gt(ind['ma200'])
            & ind['recovery'].between(0.25, 0.75)
            & ind['trough_age'].between(2, params.graph_window - 4)
            & ind['upper_bridges'].ge(params.min_bridges)
            & ind['new_upper_bridges'].ge(1)
        ).fillna(False).to_numpy(dtype=bool)
        short_entry = (
            ready & stressed & data['close'].lt(ind['ma200'])
            & ind['recovery'].between(0.25, 0.75)
            & ind['peak_age'].between(2, params.graph_window - 4)
            & ind['lower_bridges'].ge(params.min_bridges)
            & ind['new_lower_bridges'].ge(1)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # The entry open is observed only on the actual fill bar.
                if i == entry_bar:
                    entry_price = opens[i]
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                gain = position * (close[i] / entry_price - 1.0) if not invalid_entry and np.isfinite(close[i]) else np.nan
                expired = i - entry_bar + 1 >= params.max_hold
                if invalid_entry or expired or gain >= params.profit_target:
                    position = 0
                    entry_price = np.nan
                    signal[i] = 0
                    continue
            elif long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                entry_bar = i + 1
                entry_price = np.nan
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': np.ones(len(data), dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'ready': True, 'depth': 0.05, 'ma200': 99.0, 'recovery': 0.5, 'trough_age': 3, 'peak_age': 3, 'upper_bridges': 2, 'lower_bridges': 0, 'new_upper_bridges': 0, 'new_lower_bridges': 0}, index=index)
    ind.loc[index[1], 'new_upper_bridges'] = 1
    params = VisibilityRecoveryParams(max_hold=3)
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    data.loc[index[3], 'close'] = 104.0
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    data['close'] = 100.0
    data.loc[index[3], 'close'] = 96.0
    ind['ma200'] = 101.0
    ind['new_upper_bridges'] = 0
    ind['lower_bridges'] = 2
    ind.loc[index[1], 'new_lower_bridges'] = 1
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 0, -1, -1, 0, 0, 0, 0]
