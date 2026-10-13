from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TurnoverCompressionParams:
    profit_target: float = 0.04
    holding_bars: int = 18

    def __post_init__(self):
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('profit_target must be finite and positive')
        if isinstance(self.holding_bars, bool) or not isinstance(self.holding_bars, int) or self.holding_bars < 1:
            raise ValueError('holding_bars must be a positive integer')


class GeneratedStrategy(BaseStrategy[TurnoverCompressionParams]):
    strategy_id = 'gen_local-w2_1791221695'

    @classmethod
    def params_type(cls):
        return TurnoverCompressionParams

    def warmup_bars(self, params: TurnoverCompressionParams) -> int:
        # Ten-bar envelope followed by a 126-observation rank needs 135 bars.
        return 135

    def indicators(self, data: pd.DataFrame, params: TurnoverCompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['envelope_high'] = data['high'].rolling(10, min_periods=10).max()
        out['envelope_low'] = data['low'].rolling(10, min_periods=10).min()
        out['envelope_mid'] = (out['envelope_high'] + out['envelope_low']) / 2.0
        out['width'] = ((out['envelope_high'] - out['envelope_low']) / close).replace([np.inf, -np.inf], np.nan)
        out['dollar_volume'] = (close * data['volume'].where(data['volume'] >= 0)).replace([np.inf, -np.inf], np.nan)
        out['turnover'] = out['dollar_volume'].rolling(5, min_periods=5).sum()
        out['width_rank'] = out['width'].rolling(126, min_periods=126).rank(method='average', pct=True)
        out['turnover_rank'] = out['turnover'].rolling(126, min_periods=126).rank(method='average', pct=True)
        ready = out[['width', 'turnover', 'width_rank', 'turnover_rank', 'envelope_mid']].notna().all(axis=1)
        out['entry'] = (
            ready
            & (out['width'] > 0)
            & (out['turnover'] > 0)
            & (out['width_rank'] <= 0.20)
            & (out['turnover_rank'] >= 0.80)
            & (close > out['envelope_mid'])
            & (close > data['open'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: TurnoverCompressionParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        entry = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # Read the fill bar's open only once that bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0
                target_hit = valid_price and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                expired = i - entry_bar + 1 >= params.holding_bars
                if not valid_price or target_hit or expired:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    raw[i] = 1
            elif entry[i]:
                entry_bar = i + 1
                raw[i] = 1
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': [100.0, 110.0, 110.0, 110.0, 110.0, 110.0],
        'high': [101.0, 111.0, 116.0, 111.0, 111.0, 111.0],
        'low': [99.0, 109.0, 109.0, 109.0, 109.0, 109.0],
        'close': [100.0, 110.0, 115.0, 110.0, 110.0, 110.0],
        'volume': [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({'entry': [True, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, TurnoverCompressionParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    data['close'] = 110.0
    result = strategy.generate_signals(data, indicators, None, TurnoverCompressionParams(holding_bars=3))
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    assert (result.data['size'] > 0).all()
    assert not strategy.indicators(data, TurnoverCompressionParams())['entry'].any()
    assert strategy.generate_signals(data.iloc[:0], indicators.iloc[:0], None, TurnoverCompressionParams()).data.empty


if __name__ == '__main__':
    _self_check()
