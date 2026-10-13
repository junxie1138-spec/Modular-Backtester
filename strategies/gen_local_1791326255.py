from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 12
    breakout_window: int = 20
    volume_window: int = 20
    area_min: float = 0.08
    profit_target: float = 0.04
    hold_bars: int = 18

    def __post_init__(self):
        for value in (self.signature_window, self.breakout_window, self.volume_window, self.hold_bars):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Window lengths and hold_bars must be positive integers.')
        if not np.isfinite(self.area_min) or not 0 <= self.area_min < 0.5:
            raise ValueError('area_min must be in [0, 0.5).')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('profit_target must be finite and positive.')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1791326255'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(200, params.breakout_window + 1,
                       params.volume_window + params.signature_window + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] >= 0)
        out['return'] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out['volume_base'] = volume.shift(1).rolling(params.volume_window).mean()
        out['volume_increment'] = (volume / out['volume_base'].where(out['volume_base'] > 0) - 1).replace([np.inf, -np.inf], np.nan)
        x = out['return']
        y = out['volume_increment']
        valid = x.notna() & y.notna()
        x0 = x.where(valid, 0.0)
        y0 = y.where(valid, 0.0)
        out['cumulative_return'] = x0.cumsum()
        out['cumulative_volume'] = y0.cumsum()
        X = out['cumulative_return']
        Y = out['cumulative_volume']
        length = params.signature_window
        out['cross_increment'] = X.shift(1, fill_value=0.0) * y0 - Y.shift(1, fill_value=0.0) * x0
        out['window_return'] = x0.rolling(length).sum()
        out['window_volume'] = y0.rolling(length).sum()
        out['area_scale'] = x0.abs().rolling(length).sum() * y0.abs().rolling(length).sum()
        origin_x = X.shift(length, fill_value=0.0)
        origin_y = Y.shift(length, fill_value=0.0)
        out['signed_area'] = 0.5 * (out['cross_increment'].rolling(length).sum()
                                    - origin_x * out['window_volume']
                                    + origin_y * out['window_return'])
        complete = valid.astype(int).rolling(length).sum().eq(length)
        out['signature_area'] = (out['signed_area'] / out['area_scale'].where(out['area_scale'] > 0)).where(complete).replace([np.inf, -np.inf], np.nan)
        out['ma200'] = close.rolling(200).mean()
        out['prior_high'] = data['high'].shift(1).rolling(params.breakout_window).max()
        out['entry'] = (close.gt(out['ma200'])
                        & close.gt(out['prior_high'])
                        & out['signature_area'].gt(params.area_min)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_price = None
        entry_bar = -1
        for i in range(len(data)):
            if active:
                # A decision at the previous close fills at this bar's open.
                if entry_price is None:
                    entry_price = opens[i]
                    entry_bar = i
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                target_hit = (valid_entry and np.isfinite(closes[i])
                              and closes[i] >= entry_price * (1 + params.profit_target))
                expired = i - entry_bar + 1 >= params.hold_bars
                if not valid_entry or target_hit or expired:
                    active = False
                    entry_price = None
            elif entries[i]:
                active = True
                entry_price = None
            decisions[i] = int(active)
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 100.0, 'low': 100.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry': [True] + [False] * 7}, index=index)
    params = SignatureParams(hold_bars=3)
    timed = strategy.generate_signals(data, indicators, None, params).data
    assert timed['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    data.loc[index[2], 'close'] = 105.0
    targeted = strategy.generate_signals(data, indicators, None, params).data
    assert targeted['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert targeted['size'].gt(0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
