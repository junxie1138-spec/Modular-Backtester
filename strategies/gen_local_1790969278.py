from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_bars: int = 20
    percentile_bars: int = 126
    tail_fraction: float = 0.10
    setup_bars: int = 5
    profit_target: float = 0.02
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1790969278'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.ma_bars + params.percentile_bars)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.ma_bars < 2 or params.percentile_bars < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not 0.0 < params.tail_fraction < 0.5:
            raise ValueError('tail_fraction must be between zero and one half.')
        if params.setup_bars < 1 or params.max_hold_bars < 1:
            raise ValueError('Setup and holding periods must be positive.')
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError('profit_target must be finite and positive.')
        close = data['close'].where(data['close'] > 0)
        out = pd.DataFrame(index=data.index)
        out['ma'] = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out['std'] = close.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out['z'] = ((close - out['ma']) / out['std'].where(out['std'] > 0)).replace([np.inf, -np.inf], np.nan)
        history = out['z'].shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars)
        out['lower'] = history.quantile(params.tail_fraction)
        out['upper'] = history.quantile(1.0 - params.tail_fraction)
        out['price_change'] = close.diff()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        values = indicators[['z', 'lower', 'upper', 'price_change']].to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        armed = 0
        armed_bar = -1
        boundary = np.nan

        for i in range(len(data)):
            if position:
                # A decision at the previous close first fills at this open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                # Gross close-observed return from the entry open; exit fills next open.
                target_hit = valid_entry and valid_close and position * (closes[i] / entry_price - 1.0) >= params.profit_target
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if not valid_entry or target_hit or expired:
                    position = 0
                    entry_price = np.nan
                decisions[i] = position
                continue

            z, lower, upper, change = values[i]
            if not np.isfinite(values[i]).all() or not np.isfinite(closes[i]) or closes[i] <= 0:
                armed = 0
                continue

            if armed and i - armed_bar > params.setup_bars:
                armed = 0
            if armed:
                # Discard excursions that have already passed the moving average.
                if (armed == 1 and z >= 0) or (armed == -1 and z <= 0):
                    armed = 0
                elif (armed == 1 and z > boundary and change > 0) or (armed == -1 and z < boundary and change < 0):
                    position = armed
                    entry_bar = i + 1
                    entry_price = np.nan
                    armed = 0
                    decisions[i] = position
                    continue

            if not armed:
                if z < lower < 0:
                    armed, boundary, armed_bar = 1, lower, i
                elif z > upper > 0:
                    armed, boundary, armed_bar = -1, upper, i

        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    # Verify next-open entry timing and both exits in both directions.
    params = Params(max_hold_bars=3)
    index = pd.date_range('2020-01-01', periods=7)
    for side in (1, -1):
        for hit_target in (False, True):
            close = np.full(7, 100.0)
            if hit_target:
                close[2] = 100.0 * (1.0 + side * 0.03)
            data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': close, 'volume': 1000.0}, index=index)
            ind = pd.DataFrame({'z': np.array([-3.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.0]) * side, 'lower': -2.0, 'upper': 2.0, 'price_change': float(side)}, index=index)
            result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
            expected = [0, 0, side, 0, 0, 0, 0] if hit_target else [0, 0, side, side, side, 0, 0]
            assert result['signal'].tolist() == expected
            assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
