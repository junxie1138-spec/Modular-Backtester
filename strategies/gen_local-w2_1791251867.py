from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAnalogParams:
    history: int = 120
    atr_length: int = 14
    bandwidth: float = 0.65
    min_edge: float = 0.0015
    trail_atr: float = 1.2
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[GapAnalogParams]):
    strategy_id = 'gen_local-w2_1791251867'

    @classmethod
    def params_type(cls) -> type[GapAnalogParams]:
        return GapAnalogParams

    def warmup_bars(self, params: GapAnalogParams) -> int:
        return int(params.history + params.atr_length + 6)

    def indicators(self, data: pd.DataFrame, params: GapAnalogParams) -> pd.DataFrame:
        if params.history < 5 or params.atr_length < 2:
            raise ValueError('history must be >= 5 and atr_length >= 2')
        if not np.isfinite(params.bandwidth) or params.bandwidth <= 0:
            raise ValueError('bandwidth must be positive and finite')
        if not np.isfinite(params.min_edge) or params.min_edge < 0:
            raise ValueError('min_edge must be nonnegative and finite')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('trail_atr must be positive and finite')
        if params.hold_days not in (1, 2):
            raise ValueError('hold_days must be 1 or 2')

        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out.loc[previous_close.isna(), 'true_range'] = np.nan
        out['atr'] = out['true_range'].rolling(params.atr_length).mean()
        out['prior_atr'] = out['atr'].shift(1).where(lambda s: s > 0)
        out['gap'] = (data['open'] - previous_close) / out['prior_atr']
        out['gap_lag1'] = out['gap'].shift(1)
        out['gap_lag2'] = out['gap'].shift(2)
        out['absorption'] = (data['close'] - data['open']) / out['prior_atr']
        features = out[['gap', 'gap_lag1', 'gap_lag2', 'absorption']]

        weights = []
        outcomes = []
        # A lag of at least three makes each two-session outcome known before today.
        for lag in range(3, params.history + 3):
            distance = ((features - features.shift(lag)) ** 2).mean(axis=1, skipna=False)
            outcome = data['close'].shift(lag - 2) / data['open'].shift(lag - 1) - 1.0
            weight = np.exp(-distance / (2.0 * params.bandwidth ** 2))
            valid = np.isfinite(distance) & np.isfinite(outcome) & (distance <= params.bandwidth ** 2)
            weights.append(weight.where(valid, 0.0).to_numpy())
            outcomes.append(outcome.where(valid, 0.0).to_numpy())

        w = np.column_stack(weights)
        y = np.column_stack(outcomes)
        total = w.sum(axis=1)
        denominator = np.where(total > 0, total, np.nan)
        forecast = (w * y).sum(axis=1) / denominator
        second_moment = (w * y ** 2).sum(axis=1) / denominator
        out['weight_sum'] = total
        out['forecast'] = forecast
        out['dispersion'] = np.sqrt(np.maximum(second_moment - forecast ** 2, 0.0))
        squared_weights = (w ** 2).sum(axis=1)
        out['effective_matches'] = total ** 2 / np.where(squared_weights > 0, squared_weights, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapAnalogParams) -> SignalFrame:
        ready = (
            (indicators['forecast'] > params.min_edge)
            & (indicators['forecast'] > 0.2 * indicators['dispersion'])
            & (indicators['effective_matches'] >= 5.0)
            & (indicators['gap'] < 0.0)
            & (indicators['absorption'] > 0.0)
            & (indicators['atr'] > 0.0)
        ).fillna(False)
        confirmed = (ready & ready.shift(1, fill_value=False)).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        holding = False
        age = 0
        high_water = np.nan
        stop = -np.inf

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if holding:
                # Entry was ordered yesterday and filled at today's open.
                age += 1
                if not valid:
                    holding = False
                else:
                    high_water = close if np.isnan(high_water) else max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                    if close <= stop or age >= params.hold_days:
                        holding = False
                raw[i] = int(holding)
            elif valid and confirmed[i]:
                holding = True
                age = 0
                high_water = np.nan
                stop = -np.inf
                raw[i] = 1

        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapAnalogParams()
    index = pd.date_range('2020-01-01', periods=180)
    x = np.arange(len(index), dtype=float)
    close = 100.0 + 0.02 * x + np.sin(x)
    data = pd.DataFrame({'open': close - 0.2, 'high': close + 1.0,
                         'low': close - 1.0, 'close': close, 'volume': 1000.0}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:160], params)
    assert np.allclose(full.iloc[:160].to_numpy(), prefix.to_numpy(), equal_nan=True)
    small = data.iloc[:6].copy()
    small['close'] = [100.0, 100.0, 103.0, 101.0, 101.0, 102.0]
    evidence = pd.DataFrame({'forecast': 0.01, 'dispersion': 0.01,
                             'effective_matches': 10.0, 'gap': -0.5,
                             'absorption': 0.5, 'atr': 1.0}, index=small.index)
    result = strategy.generate_signals(small, evidence, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 1]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
