from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    fractional_order: float = 0.45
    fractional_window: int = 64
    correlation_window: int = 60
    echo_lag: int = 10
    correlation_floor: float = 0.25
    impulse_threshold: float = 1.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = 'gen_local-w2_1791230912'

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return int(max(200, params.fractional_window + params.echo_lag + params.correlation_window))

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError('fractional_order must be between zero and one')
        if params.fractional_window < 2 or params.correlation_window < 3 or params.echo_lag < 1 or params.holding_bars < 1:
            raise ValueError('Invalid lookback or holding period')
        if not np.isfinite(params.correlation_floor) or not -1.0 <= params.correlation_floor <= 1.0:
            raise ValueError('Invalid correlation threshold')
        if not np.isfinite(params.impulse_threshold) or params.impulse_threshold <= 0.0:
            raise ValueError('Invalid impulse threshold')
        close = data['close'].astype(float).where(data['close'] > 0.0).replace([np.inf, -np.inf], np.nan)
        fractional_close = close.copy()
        weight = 1.0
        # ponytail: finite fractional kernel; increase the window if truncation matters.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)
        impulse = fractional_close.diff()
        history = impulse.shift(1).rolling(params.correlation_window, min_periods=params.correlation_window)
        impulse_mean = history.mean()
        impulse_std = history.std(ddof=0).replace(0.0, np.nan)
        impulse_z = ((impulse - impulse_mean) / impulse_std).replace([np.inf, -np.inf], np.nan)
        lagged_impulse = impulse.shift(params.echo_lag)
        recurrence = impulse.rolling(params.correlation_window, min_periods=params.correlation_window).corr(lagged_impulse)
        echo_z = impulse_z.shift(params.echo_lag)
        regime_ma = close.rolling(200, min_periods=200).mean()
        out = pd.DataFrame({
            'fractional_close': fractional_close,
            'impulse': impulse,
            'impulse_mean': impulse_mean,
            'impulse_std': impulse_std,
            'impulse_z': impulse_z,
            'lagged_impulse': lagged_impulse,
            'recurrence': recurrence,
            'echo_z': echo_z,
            'regime_ma': regime_ma,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)
        out['entry'] = (
            out.notna().all(axis=1)
            & (close > regime_ma)
            & (recurrence > params.correlation_floor)
            & (echo_z > params.impulse_threshold)
            & (impulse_z > 0.0)
            & (impulse_z < params.impulse_threshold)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError('holding_bars must be positive')
        entry = indicators['entry'].reindex(data.index).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, eligible in enumerate(entry):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
            elif eligible:
                signal[i] = 1
                exit_bar = i + params.holding_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=24, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry': True}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, EchoParams()).data
    assert result['signal'].iloc[0] == 0
    assert result['signal'].iloc[1:9].eq(1).all()
    assert result['signal'].iloc[9] == 0
    assert result['signal'].iloc[10:18].eq(1).all()
    assert result['signal'].iloc[18] == 0
    assert result['size'].gt(0.0).all()


if __name__ == '__main__':
    _self_check()
