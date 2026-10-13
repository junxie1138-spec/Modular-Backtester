from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_lags: int = 10
    percentile_window: int = 126
    entry_percentile: float = 0.8
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local_1791295155'

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(params.atr_window + params.graph_lags + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if not (params.graph_lags >= 2 and params.percentile_window >= 2
                and params.atr_window >= 2 and 0 < params.entry_percentile < 1
                and params.stop_atr > 0 and 3 <= params.hold_days <= 5):
            raise ValueError('Invalid visibility strategy parameters')
        out = pd.DataFrame(index=data.index)
        previous = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['shock'] = (out['return'] * previous / out['atr'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out['magnitude'] = out['shock'].abs()
        out['direction'] = np.sign(out['shock'])
        weighted_agreement = pd.Series(0.0, index=data.index)
        edge_weight = pd.Series(0.0, index=data.index)
        edge_count = pd.Series(0, index=data.index)
        for lag in range(2, params.graph_lags + 1):
            prior = out['magnitude'].shift(lag)
            interior = out['magnitude'].shift(1).rolling(lag - 1).max()
            endpoints = pd.concat([out['magnitude'], prior], axis=1).min(axis=1, skipna=False)
            visible = (endpoints > interior) & endpoints.notna() & interior.notna()
            out[f'visible_{lag}'] = visible.astype(int)
            agreement = out['direction'] * out['direction'].shift(lag)
            out[f'agreement_{lag}'] = agreement.where(visible, 0.0).fillna(0.0)
            weighted_agreement += out[f'agreement_{lag}'] / lag
            edge_weight += visible.astype(float) / lag
            edge_count += visible.astype(int)
        out['edge_count'] = edge_count
        out['edge_weight'] = edge_weight
        out['assortativity'] = weighted_agreement / edge_weight.replace(0, np.nan)
        out['score'] = (out['shock'] * out['assortativity']).where(edge_count >= 2)
        out['strength'] = out['score'].abs()
        out['threshold'] = out['strength'].shift(1).rolling(
            params.percentile_window, min_periods=max(2, params.percentile_window // 3)
        ).quantile(params.entry_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        score = indicators['score'].to_numpy(dtype=float)
        threshold = indicators['threshold'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        pending_entry = False
        entry_bar = -1
        stop_price = np.nan
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if not np.isfinite(opening[i]) or opening[i] <= 0:
                    position = 0
                else:
                    entry_bar = i
                    # Freeze the stop at the next-bar entry open using decision-bar ATR.
                    stop_price = opening[i] - position * params.stop_atr * atr[i - 1]
            if position:
                stopped = np.isfinite(close[i]) and (
                    (position == 1 and close[i] < stop_price)
                    or (position == -1 and close[i] > stop_price)
                )
                expired = i - entry_bar + 1 >= params.hold_days
                if stopped or expired:
                    position = 0
                decisions[i] = position
                continue
            ready = (
                i >= self.warmup_bars(params)
                and np.isfinite(score[i]) and np.isfinite(threshold[i])
                and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(close[i]) and close[i] > 0
            )
            if ready and abs(score[i]) > threshold[i] and score[i] != 0:
                position = 1 if score[i] > 0 else -1
                pending_entry = True
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = VisibilityParams(hold_days=5)
    strategy = GeneratedStrategy()
    start = strategy.warmup_bars(params)
    n = start + 8
    index = pd.date_range('2020-01-01', periods=n, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not np.isinf(derived.to_numpy(dtype=float)).any()
    ind = pd.DataFrame({'atr': 1.0, 'score': 0.0, 'threshold': 1.0}, index=index)
    ind.iloc[start, ind.columns.get_loc('score')] = 2.0
    data.iloc[start + 2, data.columns.get_loc('close')] = 110.0
    data.iloc[start + 3, data.columns.get_loc('close')] = 101.0
    data.iloc[start + 4, data.columns.get_loc('close')] = 97.0
    signals = strategy.generate_signals(data, ind, None, params).data['signal']
    assert signals.iloc[start] == 0 and signals.iloc[start + 1] == 1
    assert signals.iloc[start + 4] == 1 and signals.iloc[start + 5] == 0
    data['close'] = 100.0
    signals = strategy.generate_signals(data, ind, None, params).data['signal']
    assert (signals.iloc[start + 1:start + 6] == 1).all()
    assert signals.iloc[start + 6] == 0
    ind.iloc[start, ind.columns.get_loc('score')] = -2.0
    data.iloc[start + 1, data.columns.get_loc('close')] = 103.0
    signals = strategy.generate_signals(data, ind, None, params).data['signal']
    assert signals.iloc[start + 1] == -1 and signals.iloc[start + 2] == 0


if __name__ == '__main__':
    _self_check()
