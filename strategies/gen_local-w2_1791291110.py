from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_window: int = 4
    volume_window: int = 20
    percentile_window: int = 126
    area_percentile: float = 0.80
    breakout_window: int = 3
    atr_window: int = 14
    atr_multiple: float = 0.80
    max_hold: int = 2

    def __post_init__(self):
        if min(self.path_window, self.volume_window, self.percentile_window,
               self.breakout_window, self.atr_window) < 2:
            raise ValueError('Lookback windows must be at least two.')
        if not 0.0 < self.area_percentile < 1.0:
            raise ValueError('area_percentile must lie strictly between zero and one.')
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError('atr_multiple must be finite and positive.')
        if self.max_hold not in (1, 2):
            raise ValueError('max_hold must be one or two sessions.')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local-w2_1791291110'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(
            params.volume_window + params.path_window + params.percentile_window + 1,
            params.atr_window + 1,
            params.breakout_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous_close = close.shift(1)
        out['return_step'] = (close / previous_close.where(previous_close > 0) - 1).replace([np.inf, -np.inf], np.nan)
        out['volume_baseline'] = data['volume'].rolling(params.volume_window).mean().shift(1)
        out['volume_step'] = (data['volume'].where(data['volume'] >= 0) / out['volume_baseline'].where(out['volume_baseline'] > 0)).replace([np.inf, -np.inf], np.nan)

        # Signed second-order area: earlier volume paired with later returns.
        area = pd.Series(0.0, index=data.index)
        for earlier in range(params.path_window - 1):
            old = params.path_window - 1 - earlier
            for later in range(earlier + 1, params.path_window):
                new = params.path_window - 1 - later
                area = area + (
                    out['volume_step'].shift(old) * out['return_step'].shift(new)
                    - out['return_step'].shift(old) * out['volume_step'].shift(new)
                )
        out['raw_area'] = area
        out['area_scale'] = (
            out['return_step'].abs().rolling(params.path_window).sum()
            * out['volume_step'].rolling(params.path_window).sum()
        )
        out['signature_area'] = (out['raw_area'] / out['area_scale'].where(out['area_scale'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['area_threshold'] = out['signature_area'].shift(1).rolling(params.percentile_window).quantile(params.area_percentile)
        out['breakout_high'] = data['high'].shift(1).rolling(params.breakout_window).max()
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['entry'] = (
            (out['signature_area'] > out['area_threshold'])
            & (close > out['breakout_high'])
            & (out['atr'] > 0)
            & np.isfinite(close)
            & np.isfinite(out['atr'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        targets = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        held = 0
        high_water = -np.inf
        stop = -np.inf

        for bar in range(len(data)):
            close, atr = closes[bar], atrs[bar]
            valid = np.isfinite(close) and np.isfinite(atr) and close > 0 and atr > 0
            if active:
                # The first held bar follows the entry decision and its next-open fill.
                held += 1
                if valid:
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.atr_multiple * atr)
                if not valid or close <= stop or held >= params.max_hold:
                    active = False
                else:
                    targets[bar] = 1
            elif valid and entries[bar]:
                active = True
                held = 0
                high_water = -np.inf
                stop = -np.inf
                targets[bar] = 1

        df = pd.DataFrame({'signal': targets, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'close': [100.0, 105.0, 106.0, 107.0, 108.0]}, index=index)
    derived = pd.DataFrame({'atr': 1.0, 'entry': [True, False, False, False, False]}, index=index)
    result = strategy.generate_signals(data, derived, None, SignatureParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    result = strategy.generate_signals(data, derived, None, SignatureParams(max_hold=1)).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 0]
    assert (result['size'] > 0).all()
