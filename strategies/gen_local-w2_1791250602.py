from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_window: int = 18
    percentile_window: int = 126
    area_percentile: float = 0.60


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791250602"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(2 * params.path_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        n = params.path_window
        h = params.percentile_window
        if n < 2 or h < 2 or not 0.0 < params.area_percentile < 1.0:
            raise ValueError("Windows must exceed one and percentile must lie inside (0, 1).")
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        volume = data['volume'].astype(float).where(data['volume'] >= 0)
        out['price_step'] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out['volume_step'] = np.log1p(volume).diff().replace([np.inf, -np.inf], np.nan)
        out['price_path'] = out['price_step'].fillna(0.0).cumsum()
        out['volume_path'] = out['volume_step'].fillna(0.0).cumsum()
        out['cross_step'] = (
            out['price_path'].shift(1) * out['volume_step']
            - out['volume_path'].shift(1) * out['price_step']
        )
        out['price_origin'] = out['price_path'].shift(n)
        out['volume_origin'] = out['volume_path'].shift(n)
        out['price_displacement'] = out['price_step'].rolling(n, min_periods=n).sum()
        out['volume_displacement'] = out['volume_step'].rolling(n, min_periods=n).sum()
        out['cross_sum'] = out['cross_step'].rolling(n, min_periods=n).sum()
        out['price_variation'] = out['price_step'].abs().rolling(n, min_periods=n).sum()
        out['volume_variation'] = out['volume_step'].abs().rolling(n, min_periods=n).sum()
        out['area_scale'] = (out['price_variation'] * out['volume_variation']).replace(0.0, np.nan)
        # Negative price-volume area means volume movement preceded price movement.
        out['lead_area'] = -0.5 * (
            out['cross_sum']
            - out['price_origin'] * out['volume_displacement']
            + out['volume_origin'] * out['price_displacement']
        ) / out['area_scale']
        out['lead_area'] = out['lead_area'].replace([np.inf, -np.inf], np.nan)
        out['smoothed_area'] = out['lead_area'].rolling(n, min_periods=n).mean()
        out['area_threshold'] = out['smoothed_area'].shift(1).rolling(h, min_periods=h).quantile(params.area_percentile)
        out['smoothed_price'] = close.rolling(n, min_periods=n).mean()
        out['breakout_ceiling'] = out['smoothed_price'].shift(1).rolling(n, min_periods=n).max()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        required = ['smoothed_area', 'area_threshold', 'smoothed_price', 'breakout_ceiling']
        ready = indicators[required].notna().all(axis=1)
        entry_condition = (
            ready
            & (indicators['smoothed_area'] > indicators['area_threshold'])
            & (indicators['smoothed_price'] > indicators['breakout_ceiling'])
        )
        df = pd.DataFrame(index=data.index)
        # Remain long while the entry predicate holds; exit only when it flips false.
        df['signal'] = entry_condition.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == '__main__':
    params = SignatureParams(path_window=4, percentile_window=8)
    t = np.arange(100, dtype=float)
    close = 100.0 * np.exp(0.003 * t + 0.008 * np.sin(t / 5.0))
    data = pd.DataFrame({
        'open': close,
        'high': close * 1.01,
        'low': close * 0.99,
        'close': close,
        'volume': 1e6 * np.exp(0.3 * np.sin(t / 5.0 + 0.8)),
    }, index=pd.date_range('2020-01-01', periods=len(t)))
    strategy = GeneratedStrategy()
    ind = strategy.indicators(data, params)
    end = 40
    dx = ind['price_step'].iloc[end - params.path_window + 1:end + 1].to_numpy()
    dy = ind['volume_step'].iloc[end - params.path_window + 1:end + 1].to_numpy()
    x_previous = np.cumsum(dx) - dx
    y_previous = np.cumsum(dy) - dy
    expected_area = -0.5 * np.sum(x_previous * dy - y_previous * dx) / (np.abs(dx).sum() * np.abs(dy).sum())
    assert np.isclose(ind['lead_area'].iloc[end], expected_area)
    signals = strategy.generate_signals(data, ind, None, params).data
    condition = ((ind['smoothed_area'] > ind['area_threshold']) & (ind['smoothed_price'] > ind['breakout_ceiling']))
    expected_signal = condition.astype(int).shift(1).fillna(0).astype(int)
    assert signals['signal'].equals(expected_signal)
    assert signals['signal'].isin([0, 1]).all() and signals['size'].gt(0).all()
    prefix = strategy.indicators(data.iloc[:70], params)
    pd.testing.assert_frame_equal(ind.iloc[:70], prefix)
