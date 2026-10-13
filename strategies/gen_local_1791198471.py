from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    analog_count: int = 12
    holding_bars: int = 18

    def __post_init__(self):
        if not isinstance(self.analog_count, int) or not 1 <= self.analog_count <= 504:
            raise ValueError('analog_count must be an integer between 1 and 504')
        if not isinstance(self.holding_bars, int) or not 15 <= self.holding_bars <= 20:
            raise ValueError('holding_bars must be an integer between 15 and 20')


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local_1791198471'

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return 504 + 25 + params.holding_bars + 1

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        opening = data['open'].where(data['open'] > 0)
        closing = data['close'].where(data['close'] > 0)
        out['gap'] = np.log(opening / closing.shift(1))
        out['intraday'] = np.log(closing / opening)
        out['gap_scale'] = out['gap'].rolling(20, min_periods=20).std().replace(0.0, np.nan)
        shape_columns = []
        gap_path = pd.Series(0.0, index=data.index)
        repair_path = pd.Series(0.0, index=data.index)
        for step in range(5):
            lag = 4 - step
            gap_path = gap_path + out['gap'].shift(lag)
            repair_path = repair_path + out['intraday'].shift(lag)
            gap_name = f'gap_path_{step}'
            repair_name = f'repair_path_{step}'
            out[gap_name] = gap_path / out['gap_scale']
            out[repair_name] = repair_path / out['gap_scale']
            shape_columns.extend([gap_name, repair_name])
        out['shock'] = out['gap'].shift(4).lt(-out['gap_scale'])
        out['eligible'] = (
            out['shock']
            & closing.gt(opening.shift(4))
            & closing.lt(closing.shift(5))
        )
        delay = params.holding_bars + 1
        mature_columns = []
        for name in shape_columns:
            mature_name = f'mature_{name}'
            out[mature_name] = out[name].shift(delay)
            mature_columns.append(mature_name)
        out['mature_shock'] = out['shock'].shift(delay).fillna(False).astype(bool)
        # At row r, this is the completed open-to-open trade from r-h to r.
        out['mature_return'] = opening / opening.shift(params.holding_bars) - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        shapes = out[shape_columns].to_numpy(dtype=float)
        library = out[mature_columns].to_numpy(dtype=float)
        outcomes = out['mature_return'].to_numpy(dtype=float)
        valid = (
            np.isfinite(library).all(axis=1)
            & np.isfinite(outcomes)
            & out['mature_shock'].to_numpy(dtype=bool)
        )
        eligible = out['eligible'].to_numpy(dtype=bool)
        forecast = np.full(len(data), np.nan)
        win_rate = np.full(len(data), np.nan)
        # ponytail: scan at most 504 matured records; index search if the archive grows.
        for i in range(self.warmup_bars(params), len(data)):
            if not eligible[i] or not np.isfinite(shapes[i]).all():
                continue
            start = max(0, i - 503)
            candidates = np.flatnonzero(valid[start:i + 1]) + start
            if candidates.size < params.analog_count:
                continue
            distance = np.square(library[candidates] - shapes[i]).sum(axis=1)
            nearest = np.argpartition(distance, params.analog_count - 1)[:params.analog_count]
            returns = outcomes[candidates[nearest]]
            forecast[i] = np.median(returns)
            win_rate[i] = np.mean(returns > 0.0)
        out['forecast'] = forecast
        out['win_rate'] = win_rate
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShapeParams,
    ) -> SignalFrame:
        entries = (
            indicators['eligible'].fillna(False)
            & indicators['forecast'].gt(0.0014)
            & indicators['win_rate'].ge(0.6)
        ).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(self.warmup_bars(params), len(data)):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    # Force a flat bar even if a fresh entry condition is present.
                    exit_at = -1
            elif entries[i]:
                signal[i] = 1
                exit_at = i + params.holding_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    for holding in (15, 18, 20):
        params = ShapeParams(holding_bars=holding)
        start = strategy.warmup_bars(params)
        index = pd.date_range('2000-01-01', periods=start + 2 * holding + 5)
        prices = pd.DataFrame(100.0, index=index, columns=['open', 'high', 'low', 'close', 'volume'])
        features = pd.DataFrame({'eligible': True, 'forecast': 0.01, 'win_rate': 0.8}, index=index)
        result = strategy.generate_signals(prices, features, None, params).data
        assert result['signal'].iloc[:start + 1].eq(0).all()
        assert result['signal'].iloc[start + 1:start + holding + 1].eq(1).all()
        assert result['signal'].iloc[start + holding + 1] == 0
        assert result['signal'].iloc[start + holding + 2] == 1
        assert set(result['signal'].unique()) <= {0, 1}
        assert result['size'].gt(0).all()
