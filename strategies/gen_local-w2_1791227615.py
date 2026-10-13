from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InventoryParams:
    peak_window: int = 21
    inventory_window: int = 10
    rank_window: int = 63
    entry_percentile: float = 0.80
    minimum_drawdown: float = 0.005
    minimum_size: float = 0.25
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[InventoryParams]):
    strategy_id = 'gen_local-w2_1791227615'

    @classmethod
    def params_type(cls) -> type[InventoryParams]:
        return InventoryParams

    def warmup_bars(self, params: InventoryParams) -> int:
        return int(params.peak_window + params.inventory_window + params.rank_window)

    def indicators(self, data: pd.DataFrame, params: InventoryParams) -> pd.DataFrame:
        if min(params.peak_window, params.inventory_window, params.rank_window) < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if not 0.5 < params.entry_percentile < 1.0:
            raise ValueError('entry_percentile must be between 0.5 and 1')
        if not 0.0 < params.minimum_drawdown < 1.0:
            raise ValueError('minimum_drawdown must be between 0 and 1')
        if not 0.0 < params.minimum_size <= params.maximum_size <= 1.0:
            raise ValueError('Position sizes must satisfy 0 < minimum <= maximum <= 1')

        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        volume = data['volume'].astype(float).replace([np.inf, -np.inf], np.nan)
        volume = volume.where(volume >= 0.0)
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        drawdown = (1.0 - close / peak).clip(lower=0.0)
        underwater = drawdown >= params.minimum_drawdown
        inventory = pd.Series(0.0, index=data.index)
        reclaimed = pd.Series(0.0, index=data.index)
        history_valid = close.notna() & volume.notna() & peak.notna()
        ready = history_valid.copy()

        for lag in range(1, params.inventory_window + 1):
            prior_close = close.shift(lag)
            prior_volume = volume.shift(lag)
            prior_underwater = underwater.shift(lag, fill_value=False)
            eligible_volume = prior_volume.where(prior_underwater, 0.0).fillna(0.0)
            inventory += eligible_volume
            reclaimed += eligible_volume.where(close >= prior_close, 0.0)
            ready &= history_valid.shift(lag, fill_value=False)

        clearance = (reclaimed / inventory.replace(0.0, np.nan)).fillna(0.5)
        clearance = clearance.where(ready)
        percentile = clearance.rolling(params.rank_window, min_periods=params.rank_window).rank(method='average', pct=True)
        ready &= percentile.notna()
        return pd.DataFrame({
            'peak': peak.fillna(0.0),
            'drawdown': drawdown.fillna(0.0),
            'inventory': inventory,
            'reclaimed': reclaimed,
            'clearance': clearance.fillna(0.5),
            'percentile': percentile.fillna(0.5),
            'ready': ready,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: InventoryParams) -> SignalFrame:
        rank = indicators['percentile'].fillna(0.5).clip(0.0, 1.0)
        ready = indicators['ready'].fillna(False).astype(bool)
        eligible = ready & (indicators['inventory'] > 0.0) & (indicators['drawdown'] >= params.minimum_drawdown)
        upper = rank >= params.entry_percentile
        lower = rank <= 1.0 - params.entry_percentile
        signals = np.zeros(len(data), dtype=int)
        position = 0

        for i in range(len(data)):
            if ready.iat[i]:
                if position == 1 and lower.iat[i]:
                    position = 0
                elif position == -1 and upper.iat[i]:
                    position = 0
                elif position == 0 and eligible.iat[i]:
                    if upper.iat[i]:
                        position = 1
                    elif lower.iat[i]:
                        position = -1
            signals[i] = position

        confidence = (2.0 * rank - 1.0).abs()
        size = params.minimum_size + (params.maximum_size - params.minimum_size) * confidence
        df = pd.DataFrame({'signal': signals, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = InventoryParams()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({
        'percentile': [0.5, 0.9, 0.6, 0.1, 0.1, 0.4, 0.9, 0.9],
        'ready': [False, True, True, True, True, True, True, True],
        'inventory': 1000.0,
        'drawdown': 0.02,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert np.isclose(result['size'].iloc[2], 0.81)
    assert result['size'].between(params.minimum_size, params.maximum_size).all()


if __name__ == '__main__':
    _self_check()
