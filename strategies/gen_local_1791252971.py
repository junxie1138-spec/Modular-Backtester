from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CycleParams:
    ma_window: int = 40
    correlation_window: int = 60
    z_entry: float = 1.4
    min_negative_correlation: float = 0.1
    forecast_entry: float = 0.15
    atr_window: int = 14
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[CycleParams]):
    strategy_id = 'gen_local_1791252971'

    @classmethod
    def params_type(cls) -> type[CycleParams]:
        return CycleParams

    def warmup_bars(self, params: CycleParams) -> int:
        return int(max(params.ma_window, params.correlation_window + 10, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: CycleParams) -> pd.DataFrame:
        if min(params.ma_window, params.correlation_window, params.atr_window) < 2:
            raise ValueError('Rolling windows must be at least two bars.')
        thresholds = np.array([params.z_entry, params.min_negative_correlation, params.forecast_entry, params.breakeven_pct, params.trail_atr], dtype=float)
        if not np.isfinite(thresholds).all() or (thresholds <= 0).any():
            raise ValueError('Thresholds must be finite and positive.')
        if params.min_negative_correlation >= 1 or params.holding_bars < 1:
            raise ValueError('Invalid correlation threshold or holding period.')
        close = data['close'].astype(float)
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = close.shift(1)
        out['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['price_std'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out['z'] = (close - out['ma']) / out['price_std'].where(out['price_std'] > 0)
        out['weekly_return'] = close.pct_change(periods=5, fill_method=None)
        out['lagged_weekly_return'] = out['weekly_return'].shift(5)
        out['weekly_correlation'] = out['weekly_return'].rolling(params.correlation_window, min_periods=params.correlation_window).corr(out['lagged_weekly_return']).clip(-1.0, 1.0)
        out['weekly_std'] = out['weekly_return'].rolling(params.correlation_window, min_periods=params.correlation_window).std(ddof=0)
        out['forecast'] = out['weekly_correlation'] * out['weekly_return'] / out['weekly_std'].where(out['weekly_std'] > 0)
        out['true_range'] = pd.concat([(data['high'] - data['low']).abs(), (data['high'] - out['previous_close']).abs(), (data['low'] - out['previous_close']).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CycleParams) -> SignalFrame:
        required = indicators[['z', 'weekly_correlation', 'forecast', 'atr']]
        valid = pd.Series(np.isfinite(required.to_numpy(dtype=float)).all(axis=1), index=data.index) & indicators['atr'].gt(0)
        alternating = indicators['weekly_correlation'].le(-params.min_negative_correlation)
        long_entry = valid & alternating & indicators['z'].le(-params.z_entry) & indicators['forecast'].ge(params.forecast_entry)
        short_entry = valid & alternating & indicators['z'].ge(params.z_entry) & indicators['forecast'].le(-params.forecast_entry)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        side = 0
        entry_price = 0.0
        entry_bar = -1
        stop = -np.inf
        armed = False
        pending = False
        for i in range(len(data)):
            if side:
                if pending:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        pending = False
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                    stop = side * entry_price - params.trail_atr * atr[i - 1]
                    armed = False
                    pending = False
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    desired[i] = side
                    continue
                # Direction-adjusted stop increases for either side; a short's raw-price stop decreases.
                price = side * closes[i]
                expired = i - entry_bar + 1 >= params.holding_bars
                if price <= stop or expired:
                    side = 0
                    continue
                profit = side * (closes[i] / entry_price - 1.0)
                if profit >= params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, side * entry_price)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, price - params.trail_atr * atr[i])
                desired[i] = side
            elif candidates[i] and (i == 0 or candidates[i] != candidates[i - 1]):
                side = int(candidates[i])
                pending = True
                desired[i] = side
        # Stops are evaluated at the close and execute through the next-bar signal.
        df = pd.DataFrame({'signal': desired, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = CycleParams()
    index = pd.date_range('2020-01-01', periods=5, freq='B')
    for direction in (1, -1):
        close = np.array([100.0, 104.0, 103.0, 100.5, 100.0]) if direction == 1 else np.array([100.0, 96.0, 97.0, 99.5, 100.0])
        data = pd.DataFrame({'open': 100.0, 'high': close + 1.0, 'low': close - 1.0, 'close': close, 'volume': 1000.0}, index=index)
        indicators = pd.DataFrame({'z': [-2.0 * direction, 0.0, 0.0, 0.0, 0.0], 'weekly_correlation': -0.4, 'forecast': [float(direction), 0.0, 0.0, 0.0, 0.0], 'atr': 1.0}, index=index)
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, direction, 0]
        assert result['size'].gt(0).all()
    flat = data.copy()
    flat[['open', 'high', 'low', 'close']] = 100.0
    derived = GeneratedStrategy.indicators(None, flat, params)
    result = GeneratedStrategy.generate_signals(None, flat, derived, None, params).data
    assert result['signal'].eq(0).all()
    assert GeneratedStrategy.warmup_bars(None, params) >= params.correlation_window + 10


if __name__ == '__main__':
    _self_check()
