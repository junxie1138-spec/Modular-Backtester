from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_window: int = 6
    area_threshold: float = 0.08
    autocorr_window: int = 40
    autocorr_ceiling: float = -0.05
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local-w2_1791342373'

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return int(max(200, params.area_window + 1, params.autocorr_window + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.area_window < 2 or params.autocorr_window < 3:
            raise ValueError('Indicator windows are too short')
        if not np.isfinite(params.area_threshold) or not 0 < params.area_threshold < 0.5:
            raise ValueError('area_threshold must be between zero and 0.5')
        if not np.isfinite(params.autocorr_ceiling) or not -1 < params.autocorr_ceiling < 0:
            raise ValueError('autocorr_ceiling must be between -1 and zero')
        if params.holding_bars not in (3, 4, 5):
            raise ValueError('holding_bars must be 3, 4, or 5')
        out = pd.DataFrame(index=data.index)
        out['return'] = np.log(data['close'].where(data['close'] > 0)).diff()
        out['variation'] = out['return'].abs()
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature: sum over ordered increment pairs.
        for lag in range(1, params.area_window):
            cross = (
                out['return'].shift(lag) * out['variation']
                - out['variation'].shift(lag) * out['return']
            )
            width = params.area_window - lag
            area = area + 0.5 * cross.rolling(width, min_periods=width).sum()
        out['signature_area'] = area
        out['total_variation'] = out['variation'].rolling(
            params.area_window, min_periods=params.area_window
        ).sum()
        out['normalized_area'] = area / out['total_variation'].pow(2).replace(0.0, np.nan)
        out['lagged_return'] = out['return'].shift(1)
        out['autocorrelation'] = out['return'].rolling(
            params.autocorr_window, min_periods=params.autocorr_window
        ).corr(out['lagged_return'])
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        if params.holding_bars not in (3, 4, 5):
            raise ValueError('holding_bars must be 3, 4, or 5')
        ready = indicators[['normalized_area', 'autocorrelation', 'ma200']].notna().all(axis=1)
        eligible = ready & (indicators['autocorrelation'] <= params.autocorr_ceiling)
        long_entry = eligible & (indicators['normalized_area'] <= -params.area_threshold) & (data['close'] > indicators['ma200'])
        short_entry = eligible & (indicators['normalized_area'] >= params.area_threshold) & (data['close'] < indicators['ma200'])
        entries = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, entry in enumerate(entries):
            if position != 0 and bar == exit_bar:
                position = 0
                # Reserve the expiry bar for an explicit flat target.
            elif position == 0 and entry != 0:
                position = int(entry)
                exit_bar = bar + params.holding_bars
            signal[bar] = position
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = SignatureParams()
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'normalized_area': -0.2, 'autocorrelation': -0.3, 'ma200': 90.0}, index=index)
    # Reverse all entry conditions after entry; the fixed expiry must remain unchanged.
    indicators.iloc[1:5] = [0.2, 0.3, 110.0]
    indicators.iloc[5:] = [0.2, -0.3, 110.0]
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, -1]
    assert result['size'].gt(0).all()
    calculated = GeneratedStrategy.indicators(data, params)
    assert calculated.index.equals(data.index)
    assert calculated['normalized_area'].isna().all()
    assert GeneratedStrategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
