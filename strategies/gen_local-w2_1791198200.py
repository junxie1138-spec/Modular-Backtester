from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    slope_window: int = 20
    revision_window: int = 6
    percentile_window: int = 126
    entry_percentile: float = 0.85
    profit_target: float = 0.04
    max_hold: int = 18

    def __post_init__(self):
        if min(self.slope_window, self.revision_window, self.percentile_window) < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must be between zero and one.')
        if self.profit_target <= 0.0 or self.max_hold < 1:
            raise ValueError('Profit target and holding period must be positive.')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1791198200'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.slope_window + params.revision_window + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        out['return'] = out['log_close'].diff()
        out['noise'] = out['return'].rolling(params.slope_window, min_periods=params.slope_window).std(ddof=0)
        window = params.slope_window
        weights = np.arange(window, dtype=float) - (window - 1) / 2.0
        weights /= np.dot(weights, weights)
        out['slope'] = np.nan
        if len(data) >= window:
            out.iloc[window - 1:, out.columns.get_loc('slope')] = np.convolve(out['log_close'].to_numpy(), weights[::-1], mode='valid')
        out['innovation'] = out['slope'].diff()
        out['revision_sum'] = out['innovation'].rolling(params.revision_window, min_periods=params.revision_window).sum()
        out['revision_mass'] = out['innovation'].abs().rolling(params.revision_window, min_periods=params.revision_window).sum()
        out['coherence'] = out['revision_sum'] / out['revision_mass'].replace(0.0, np.nan)
        out['strength'] = out['slope'].abs() / out['noise'].replace(0.0, np.nan)
        out['score'] = (out['coherence'] * out['strength']).replace([np.inf, -np.inf], np.nan)
        out['threshold'] = out['score'].abs().rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.entry_percentile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        score = indicators['score'].to_numpy(dtype=float)
        threshold = indicators['threshold'].to_numpy(dtype=float)
        slope = indicators['slope'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        hot = np.isfinite(score) & np.isfinite(threshold) & (np.abs(score) > threshold)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        side = 0
                if side:
                    gain = side * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                        side = 0
            elif i + 1 < len(data) and hot[i] and (i == 0 or not hot[i - 1]):
                if np.isfinite(slope[i]) and score[i] * slope[i] > 0.0:
                    side = 1 if score[i] > 0.0 else -1
                    entry_bar = i + 1
                    entry_price = np.nan
            decisions[i] = side
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame(100.0, index=index, columns=['open', 'high', 'low', 'close', 'volume'])
    for side in (1, -1):
        ind = pd.DataFrame({'score': [0, 0, side, side, side, side, side, side], 'threshold': 0.5, 'slope': float(side)}, index=index)
        params = StrategyParams(max_hold=3, profit_target=0.03)
        result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
        assert result['signal'].tolist() == [0, 0, 0, side, side, side, 0, 0]
        assert result['size'].gt(0).all()
        target_data = data.copy()
        target_data.loc[index[4], 'close'] = 100.0 * (1.0 + side * 0.04)
        params = StrategyParams(max_hold=18, profit_target=0.03)
        result = GeneratedStrategy.generate_signals(None, target_data, ind, None, params).data
        assert result['signal'].tolist() == [0, 0, 0, side, side, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
