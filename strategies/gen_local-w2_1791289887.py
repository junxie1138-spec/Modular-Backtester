from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.6
    fractional_window: int = 64
    baseline_window: int = 40
    cusum_threshold: float = 4.5
    cusum_drift: float = 0.35
    profit_target: float = 0.035
    holding_bars: int = 18

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must lie between zero and one')
        if self.fractional_window < 2 or self.baseline_window < 2:
            raise ValueError('Indicator windows must be at least two')
        if not np.isfinite(self.cusum_threshold) or self.cusum_threshold <= 0:
            raise ValueError('cusum_threshold must be finite and positive')
        if not np.isfinite(self.cusum_drift) or self.cusum_drift < 0:
            raise ValueError('cusum_drift must be finite and nonnegative')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('profit_target must be finite and positive')
        if self.holding_bars < 1:
            raise ValueError('holding_bars must be positive')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791289887'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.fractional_window + params.baseline_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data['close'].astype(float).where(data['close'] > 0).replace([np.inf, -np.inf], np.nan)
        ma200 = close.rolling(200, min_periods=200).mean()
        fractional_close = close.copy()
        weight = 1.0
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1 - params.fractional_order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)
        innovation = fractional_close.diff()
        baseline_mean = innovation.rolling(params.baseline_window, min_periods=params.baseline_window).mean().shift(1)
        baseline_std = innovation.rolling(params.baseline_window, min_periods=params.baseline_window).std(ddof=0).shift(1)
        surprise = ((innovation - baseline_mean) / baseline_std.where(baseline_std > 1e-12)).replace([np.inf, -np.inf], np.nan)
        upward = np.zeros(len(data), dtype=float)
        downward = np.zeros(len(data), dtype=float)
        crossing = np.zeros(len(data), dtype=int)
        up = down = 0.0
        for i, value in enumerate(surprise.to_numpy()):
            if not np.isfinite(value):
                up = down = 0.0
                continue
            up = max(0.0, up + value - params.cusum_drift)
            down = max(0.0, down - value - params.cusum_drift)
            upward[i], downward[i] = up, down
            if up >= params.cusum_threshold:
                crossing[i] = 1
                up = down = 0.0
            elif down >= params.cusum_threshold:
                crossing[i] = -1
                up = down = 0.0
        entry = np.where((crossing == 1) & (close > ma200), 1,
                         np.where((crossing == -1) & (close < ma200), -1, 0))
        return pd.DataFrame({
            'ma200': ma200,
            'fractional_close': fractional_close,
            'innovation': innovation,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'surprise': surprise,
            'cusum_up': upward,
            'cusum_down': downward,
            'crossing': crossing,
            'entry': entry,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # The preceding close requested entry; this bar supplies its fill reference.
                if i == entry_bar:
                    entry_price = opens[i]
                held_bars = i - entry_bar + 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry and valid_close else np.nan
                if not valid_entry or held_bars >= params.holding_bars or gain >= params.profit_target:
                    position = 0
                    entry_price = np.nan
                signals[i] = position
                continue
            if entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=224, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 105.0, 'low': 95.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = Params(profit_target=0.03, holding_bars=3)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived['entry'].eq(0).all()
    assert strategy.warmup_bars(params) >= 200
    entries = pd.DataFrame({'entry': 0}, index=index)
    entries.iloc[200, 0] = 1
    entries.iloc[205, 0] = -1
    entries.iloc[210, 0] = 1
    data.iloc[201, data.columns.get_loc('close')] = 104.0
    data.iloc[206, data.columns.get_loc('close')] = 96.0
    result = strategy.generate_signals(data, entries, None, params).data
    assert result['signal'].iloc[200] == 0
    assert result['signal'].iloc[201] == 1 and result['signal'].iloc[202] == 0
    assert result['signal'].iloc[206] == -1 and result['signal'].iloc[207] == 0
    assert result['signal'].iloc[211:214].eq(1).all()
    assert result['signal'].iloc[214] == 0
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
