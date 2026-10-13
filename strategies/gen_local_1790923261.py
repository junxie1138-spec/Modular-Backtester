from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_window: int = 8
    reference_window: int = 30
    shock_threshold: float = 0.8

    def __post_init__(self):
        if self.path_window < 2 or self.reference_window < 5:
            raise ValueError('Lookback windows are too short')
        if not np.isfinite(self.shock_threshold) or self.shock_threshold <= 0:
            raise ValueError('shock_threshold must be finite and positive')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1790923261'

    @classmethod
    def params_type(cls):
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return int(max(200, 2 * params.path_window + params.reference_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        n = params.path_window
        m = params.reference_window
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out['volume_clock'] = (data['volume'].where(data['volume'] >= 0) / 1_000_000.0).replace([np.inf, -np.inf], np.nan)
        out['cumulative_return'] = out['return'].fillna(0.0).cumsum()
        out['cumulative_volume'] = out['volume_clock'].fillna(0.0).cumsum()
        out['origin_return'] = out['cumulative_return'].shift(n)
        out['origin_volume'] = out['cumulative_volume'].shift(n)
        out['cross_increment'] = (
            out['cumulative_return'].shift(1, fill_value=0.0) * out['volume_clock']
            - out['cumulative_volume'].shift(1, fill_value=0.0) * out['return']
        )
        out['return_total'] = out['return'].rolling(n, min_periods=n).sum()
        out['volume_total'] = out['volume_clock'].rolling(n, min_periods=n).sum()
        out['absolute_return_total'] = out['return'].abs().rolling(n, min_periods=n).sum()
        out['cross_total'] = out['cross_increment'].rolling(n, min_periods=n).sum()
        # Translate each window to its own origin: second-order signed path area.
        out['raw_area'] = 0.5 * (
            out['cross_total']
            - out['origin_return'] * out['volume_total']
            + out['origin_volume'] * out['return_total']
        )
        out['area_scale'] = out['volume_total'] * out['absolute_return_total']
        out['signature_area'] = out['raw_area'] / out['area_scale'].where(out['area_scale'] > 0)
        # The reference paths end before the current path begins.
        out['reference_area'] = out['signature_area'].shift(n)
        out['reference_mean'] = out['reference_area'].rolling(m, min_periods=m).mean()
        out['reference_std'] = out['reference_area'].rolling(m, min_periods=m).std(ddof=0)
        out['shock_z'] = (
            (out['signature_area'] - out['reference_mean'])
            / out['reference_std'].where(out['reference_std'] > 1e-12)
        )
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        area = indicators['signature_area']
        z = indicators['shock_z']
        valid = area.notna() & z.notna()
        bullish = valid & (area < 0) & (z <= -params.shock_threshold)
        bearish = valid & (area > 0) & (z >= params.shock_threshold)
        direction = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        long_ok = (data['close'] > indicators['ma200']).fillna(False).to_numpy()
        short_ok = (data['close'] < indicators['ma200']).fillna(False).to_numpy()
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(direction):
            # Opposite entry evidence is the sole exit; regime gates apply to entries.
            if position != 0 and candidate == -position:
                position = 0
            elif position == 0:
                if candidate == 1 and long_ok[i]:
                    position = 1
                elif candidate == -1 and short_ok[i]:
                    position = -1
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = positions
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = SignatureParams()
    index = pd.date_range('2020-01-01', periods=220, freq='B')
    returns = np.resize(np.array([0.01, -0.02, 0.03, 0.005, -0.01]), len(index))
    close = 100.0 * np.exp(np.cumsum(returns))
    volume = np.resize(np.array([2., 1., 4., 3., 5.]), len(index)) * 1_000_000.0
    data = pd.DataFrame({'open': close, 'high': close * 1.01, 'low': close * 0.99, 'close': close, 'volume': volume}, index=index)
    ind = GeneratedStrategy.indicators(data, params)
    r = ind['return'].iloc[-params.path_window:].to_numpy()
    v = ind['volume_clock'].iloc[-params.path_window:].to_numpy()
    expected = 0.5 * np.sum((np.cumsum(r) - r) * v - (np.cumsum(v) - v) * r)
    assert np.isclose(ind['raw_area'].iloc[-1], expected)
    assert GeneratedStrategy.warmup_bars(params) >= 200
    small = data.iloc[:6].copy()
    small['close'] = [110., 110., 110., 90., 90., 90.]
    evidence = pd.DataFrame({'signature_area': [-0.2, 0., 0.2, 0.2, 0., -0.2], 'shock_z': [-2., 0., 2., 2., 0., -2.], 'ma200': 100.}, index=small.index)
    result = GeneratedStrategy.generate_signals(small, evidence, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
