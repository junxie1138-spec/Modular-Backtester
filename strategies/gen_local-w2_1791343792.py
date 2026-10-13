from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ConditionalReturnParams:
    lookback: int = 60
    forecast_threshold: float = 0.12
    body_threshold: float = 0.20
    atr_period: int = 14
    stop_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[ConditionalReturnParams]):
    strategy_id = 'gen_local-w2_1791343792'

    @classmethod
    def params_type(cls) -> type[ConditionalReturnParams]:
        return ConditionalReturnParams

    def warmup_bars(self, params: ConditionalReturnParams) -> int:
        return int(max(max(16, int(params.lookback)) + 2, max(1, int(params.atr_period)) + 1))

    def indicators(self, data: pd.DataFrame, params: ConditionalReturnParams) -> pd.DataFrame:
        window = max(16, int(params.lookback))
        atr_window = max(1, int(params.atr_period))
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['lag_return'] = out['return'].shift(1)
        out['cross_product'] = out['return'] * out['lag_return']
        out['lag_squared'] = out['lag_return'].pow(2)
        out['pair_count'] = out['cross_product'].rolling(window, min_periods=window).count()
        for name, mask in (('positive', out['lag_return'] > 0), ('negative', out['lag_return'] < 0)):
            out[name + '_count'] = mask.astype(float).rolling(window, min_periods=window).sum()
            out[name + '_cross_sum'] = out['cross_product'].where(mask, 0.0).rolling(window, min_periods=window).sum()
            out[name + '_square_sum'] = out['lag_squared'].where(mask, 0.0).rolling(window, min_periods=window).sum()
            out[name + '_beta'] = (out[name + '_cross_sum'] / out[name + '_square_sum'].replace(0.0, np.nan)).where((out[name + '_count'] >= 8) & (out['pair_count'] >= window))
        out['selected_beta'] = out['positive_beta'].where(out['return'] > 0, out['negative_beta']).where(out['return'] != 0)
        out['return_std'] = out['return'].rolling(window, min_periods=window).std(ddof=0)
        out['forecast'] = out['selected_beta'] * out['return']
        out['forecast_score'] = out['forecast'] / out['return_std'].replace(0.0, np.nan)
        out['bar_range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['body_score'] = (data['close'] - data['open']) / out['bar_range']
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([(data['high'] - data['low']), (data['high'] - out['previous_close']).abs(), (data['low'] - out['previous_close']).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(atr_window, min_periods=atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ConditionalReturnParams) -> SignalFrame:
        forecast_threshold = max(0.0, float(params.forecast_threshold))
        body_threshold = max(0.0, float(params.body_threshold))
        forecast = indicators['forecast_score']
        body = indicators['body_score']
        candidate = np.select([(forecast > forecast_threshold) & (body > body_threshold), (forecast < -forecast_threshold) & (body < -body_threshold)], [1, -1], default=0)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        hold = min(5, max(3, int(params.hold_days)))
        stop = max(0.01, float(params.stop_atr))
        raw = np.zeros(len(data), dtype=int)
        side, age, entry, distance = 0, 0, np.nan, 0.0
        for i in range(len(data)):
            if side:
                # The preceding close scheduled entry at this bar's open.
                if np.isnan(entry):
                    entry = opens[i]
                age += 1
                # Entry price and decision-bar ATR remain fixed throughout.
                if not np.isfinite(entry) or not np.isfinite(closes[i]) or side * (closes[i] - entry) < -distance or age >= hold:
                    side, age, entry = 0, 0, np.nan
                    continue
                raw[i] = side
            elif candidate[i] and np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(candidate[i])
                entry, age, distance = np.nan, 0, stop * atr[i]
                raw[i] = side
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'forecast_score': [1.0, 0, 0, 0, 0, 0, 0], 'body_score': [1.0, 0, 0, 0, 0, 0, 0], 'atr': 1.0}, index=index)
    params = ConditionalReturnParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    data.loc[index[1], 'open'] = 103.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 0, 0, 0]
    indicators.loc[index[0], ['forecast_score', 'body_score']] = -1.0
    data.loc[index[1], 'open'] = 97.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, -1, 0, 0, 0, 0, 0]
    indicators.loc[index[0], 'body_score'] = 1.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert not result['signal'].any()
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
