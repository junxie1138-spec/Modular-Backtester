from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_bars: int = 8
    history_bars: int = 126
    area_percentile: float = 0.8
    holding_bars: int = 8

    def __post_init__(self):
        for name in ('signature_bars', 'history_bars', 'holding_bars'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{name} must be an integer')
        if self.signature_bars < 2 or self.history_bars < 2 or self.holding_bars < 1:
            raise ValueError('Invalid window or holding length')
        if not 0.5 < self.area_percentile < 1.0:
            raise ValueError('area_percentile must be between 0.5 and 1')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1791217264'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return params.signature_bars + params.history_bars + 1

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        length = params.signature_bars
        opening = data['open'].where((data['open'] > 0) & np.isfinite(data['open']))
        closing = data['close'].where((data['close'] > 0) & np.isfinite(data['close']))
        out['gap'] = np.log(opening / closing.shift(1)).replace([np.inf, -np.inf], np.nan)
        out['intraday'] = np.log(closing / opening).replace([np.inf, -np.inf], np.nan)
        out['return'] = out['gap'] + out['intraday']
        out['window_return'] = out['return'].rolling(length, min_periods=length).sum()
        out['path_length'] = out['return'].abs().rolling(length, min_periods=length).sum()
        out['efficiency'] = out['window_return'].abs() / out['path_length'].replace(0.0, np.nan)
        out['gap_activity'] = out['gap'].abs().rolling(length, min_periods=length).sum()
        out['intraday_activity'] = out['intraday'].abs().rolling(length, min_periods=length).sum()
        out['signature_area'] = 0.0
        for lag in range(1, length):
            column = f'cross_{lag}'
            out[column] = (
                out['gap'].shift(lag) * out['intraday']
                - out['intraday'].shift(lag) * out['gap']
            ).rolling(length - lag, min_periods=length - lag).sum()
            out['signature_area'] = out['signature_area'] + 0.5 * out[column]
        out['area_scale'] = out['gap_activity'] * out['intraday_activity']
        out['area'] = out['signature_area'] / out['area_scale'].replace(0.0, np.nan)
        history = params.history_bars
        prior_area = out['area'].shift(1).rolling(history, min_periods=history)
        out['area_upper'] = prior_area.quantile(params.area_percentile)
        out['area_lower'] = prior_area.quantile(1.0 - params.area_percentile)
        out['efficiency_cut'] = out['efficiency'].shift(1).rolling(history, min_periods=history).quantile(0.5)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        required = ['area', 'area_upper', 'area_lower', 'efficiency', 'efficiency_cut', 'window_return', 'intraday']
        ready = indicators[required].notna().all(axis=1)
        continuation = (
            (indicators['efficiency'] > indicators['efficiency_cut'])
            & (indicators['window_return'] > 0.0)
            & (indicators['area'] > indicators['area_upper'])
        )
        recovery = (
            (indicators['efficiency'] <= indicators['efficiency_cut'])
            & (indicators['window_return'] < 0.0)
            & (indicators['intraday'] > 0.0)
            & (indicators['area'] < indicators['area_lower'])
        )
        entries = (ready & (continuation | recovery)).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(len(data)):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    exit_at = -1
            elif entries[i]:
                signal[i] = 1
                exit_at = i + params.holding_bars
        df = pd.DataFrame(index=data.index)
        df['signal'] = signal
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = SignatureParams()
    length = 2 * params.holding_bars + 4
    index = pd.date_range('2020-01-01', periods=length)
    data = pd.DataFrame(100.0, index=index, columns=['open', 'high', 'low', 'close', 'volume'])
    indicators = pd.DataFrame({
        'area': 1.0, 'area_upper': 0.5, 'area_lower': -0.5,
        'efficiency': 1.0, 'efficiency_cut': 0.5,
        'window_return': 0.1, 'intraday': 0.01,
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    signal = strategy.generate_signals(data, indicators, ctx, params).data['signal']
    n = params.holding_bars
    assert signal.iloc[0] == 0
    assert signal.iloc[1:n + 1].eq(1).all()
    assert signal.iloc[n + 1] == 0
    assert signal.iloc[n + 2:2 * n + 2].eq(1).all()
    assert signal.iloc[2 * n + 2] == 0
    assert signal.isin([0, 1]).all()


if __name__ == '__main__':
    _self_check()
