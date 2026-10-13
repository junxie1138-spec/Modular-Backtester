from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    vol_window: int = 10
    shape_stride: int = 3
    rank_window: int = 126
    history: int = 128
    neighbors: int = 7
    horizon: int = 8
    threshold_window: int = 126
    entry_percentile: float = 0.75


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local_1791247846'

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        first_lag = params.horizon + 2 * params.shape_stride + params.vol_window
        last_lag = first_lag + params.history - 1
        return int(params.vol_window + params.rank_window + last_lag + params.threshold_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        windows = (params.vol_window, params.shape_stride, params.rank_window,
                   params.history, params.neighbors, params.horizon,
                   params.threshold_window)
        if any(v < 1 for v in windows) or params.vol_window < 2:
            raise ValueError('Lookbacks must be positive and vol_window must be at least two.')
        if params.neighbors > params.history or not 0 < params.entry_percentile < 1:
            raise ValueError('Invalid neighbor count or entry percentile.')

        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['log_return'] = np.log(close).diff()
        out['volatility'] = out['log_return'].rolling(params.vol_window).std(ddof=0).replace(0.0, np.nan)
        out['vol_percentile'] = out['volatility'].rolling(params.rank_window).rank(pct=True)
        log_vol = np.log(out['volatility'])
        out['recent_expansion'] = log_vol - log_vol.shift(params.shape_stride)
        out['older_expansion'] = log_vol.shift(params.shape_stride) - log_vol.shift(2 * params.shape_stride)
        out['matured_return'] = np.log(close / close.shift(params.horizon))
        features = out[['recent_expansion', 'older_expansion', 'vol_percentile']]
        n = len(data)
        distances = np.full((n, params.history), np.inf)
        outcomes = np.full((n, params.history), np.nan)
        first_lag = params.horizon + 2 * params.shape_stride + params.vol_window

        # ponytail: bounded dense analog search; batch rows if intraday data makes memory material.
        for column, lag in enumerate(range(first_lag, first_lag + params.history)):
            distance = ((features - features.shift(lag)) ** 2).sum(axis=1, min_count=3)
            outcome = out['matured_return'].shift(lag - params.horizon) / (
                out['volatility'].shift(lag) * np.sqrt(params.horizon)
            )
            valid = np.isfinite(distance.to_numpy()) & np.isfinite(outcome.to_numpy())
            distances[:, column] = np.where(valid, distance.to_numpy(), np.inf)
            outcomes[:, column] = np.where(valid, outcome.to_numpy(), np.nan)

        forecast = np.full(n, np.nan)
        ready = np.isfinite(distances).all(axis=1)
        if ready.any():
            selection = np.argpartition(distances[ready], params.neighbors - 1, axis=1)[:, :params.neighbors]
            matches = np.take_along_axis(outcomes[ready], selection, axis=1)
            forecast[ready] = matches.mean(axis=1)
        out['forecast'] = forecast
        out['entry_threshold'] = out['forecast'].abs().shift(1).rolling(
            params.threshold_window
        ).quantile(params.entry_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        forecast = indicators['forecast']
        threshold = indicators['entry_threshold']
        eligible = forecast.notna() & threshold.notna() & forecast.abs().gt(threshold)
        entries = np.where(eligible & forecast.gt(0), 1,
                           np.where(eligible & forecast.lt(0), -1, 0))
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for bar, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                # Opposite qualifying entry closes the position; no timed exit.
                position = 0
            positions[bar] = position
        df = pd.DataFrame({'signal': positions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = AnalogParams()
    index = pd.date_range('2020-01-01', periods=700, freq='B')
    rng = np.random.default_rng(17)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(index))))
    data = pd.DataFrame({'open': close, 'high': close * 1.01,
                         'low': close * 0.99, 'close': close,
                         'volume': 1000000.0}, index=index)
    full = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:600], params)
    assert np.allclose(full.iloc[:600].to_numpy(), prefix.to_numpy(), equal_nan=True)
    probe = pd.DataFrame({'forecast': [2.0, 0.0, -2.0, -2.0, 0.0, 2.0],
                          'entry_threshold': 1.0}, index=index[:6])
    result = GeneratedStrategy.generate_signals(data.iloc[:6], probe, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1]
    assert result['size'].gt(0).all()
