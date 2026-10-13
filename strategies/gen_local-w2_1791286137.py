from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RegimeParams:
    ma_window: int = 60
    segment_bars: int = 6
    change_threshold: float = 2.5
    displacement_threshold: float = 0.5
    efficiency_floor: float = 0.2
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[RegimeParams]):
    strategy_id = 'gen_local-w2_1791286137'

    @classmethod
    def params_type(cls) -> type[RegimeParams]:
        return RegimeParams

    def warmup_bars(self, params: RegimeParams) -> int:
        return int(max(params.ma_window + 2 * params.segment_bars + 1,
                       2 * params.segment_bars + 1))

    def indicators(self, data: pd.DataFrame, params: RegimeParams) -> pd.DataFrame:
        if params.ma_window < 2 or params.segment_bars < 2 or params.hold_bars < 1:
            raise ValueError('Windows must be at least two and hold_bars positive.')
        if not np.isfinite(params.change_threshold) or params.change_threshold <= 0:
            raise ValueError('change_threshold must be finite and positive.')
        if not np.isfinite(params.displacement_threshold) or params.displacement_threshold < 0:
            raise ValueError('displacement_threshold must be finite and nonnegative.')
        if not np.isfinite(params.efficiency_floor) or not 0 <= params.efficiency_floor <= 1:
            raise ValueError('efficiency_floor must be between zero and one.')
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        w = params.segment_bars
        out = pd.DataFrame(index=data.index)
        reference = close.rolling(params.ma_window, min_periods=params.ma_window)
        out['ma'] = reference.mean().shift(1)
        out['price_std'] = reference.std(ddof=0).shift(1)
        out['z'] = (close - out['ma']) / out['price_std'].where(out['price_std'] > 0)
        blocks = out['z'].rolling(w, min_periods=w)
        out['recent_mean'] = blocks.mean()
        out['previous_mean'] = out['recent_mean'].shift(w)
        out['recent_variance'] = blocks.var(ddof=0)
        out['previous_variance'] = out['recent_variance'].shift(w)
        out['contrast_scale'] = np.sqrt(
            (out['recent_variance'] + out['previous_variance']) / w
        ).clip(lower=0.25)
        out['change'] = (out['recent_mean'] - out['previous_mean']) / out['contrast_scale']
        out['price_step'] = close.diff()
        steps = out['price_step'].rolling(2 * w, min_periods=2 * w)
        out['net_move'] = steps.sum()
        out['path_length'] = out['price_step'].abs().rolling(2 * w, min_periods=2 * w).sum()
        out['efficiency'] = out['net_move'].abs() / out['path_length'].where(out['path_length'] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RegimeParams) -> SignalFrame:
        change = indicators['change']
        direction = np.sign(change.fillna(0))
        qualified = (
            (change.abs() >= params.change_threshold)
            & (indicators['recent_mean'] * indicators['previous_mean'] < 0)
            & (direction * indicators['z'] >= params.displacement_threshold)
            & (direction * indicators['net_move'] > 0)
            & (indicators['efficiency'] >= params.efficiency_floor)
        ).fillna(False)
        event = qualified & ~qualified.shift(1, fill_value=False)
        candidate = direction.where(event, 0).astype(int).to_numpy()
        strength = (
            0.25 + 0.70 * ((change.abs() - params.change_threshold)
                           / params.change_threshold).clip(lower=0, upper=1)
        ).fillna(0.25).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.25, dtype=float)
        side, expiry, frozen_size = 0, -1, 0.25
        for i in range(len(data)):
            if side and i == expiry:
                side = 0
            elif side == 0 and candidate[i] != 0:
                side = int(candidate[i])
                expiry = i + params.hold_bars
                frozen_size = float(strength[i])
            signal[i] = side
            size[i] = frozen_size if side else 0.25
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = RegimeParams()
    index = pd.date_range('2020-01-01', periods=24, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'change': 0.0, 'recent_mean': 0.0, 'previous_mean': 0.0,
        'z': 0.0, 'net_move': 0.0, 'efficiency': 0.0
    }, index=index)
    for i, side in [(0, 1), (3, -1), (19, -1)]:
        indicators.iloc[i] = [5.0 * side, float(side), -float(side),
                              float(side), float(side), 0.8]
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[0] == 0
    assert (result['signal'].iloc[1:19] == 1).all()
    assert result['signal'].iloc[19] == 0
    assert (result['signal'].iloc[20:] == -1).all()
    assert np.allclose(result['size'].iloc[1:19], 0.95)
    assert (result['size'] > 0).all()
    constant = pd.DataFrame({name: np.full(100, 100.0)
                             for name in ['open', 'high', 'low', 'close', 'volume']},
                            index=pd.date_range('2021-01-01', periods=100, freq='B'))
    flat = strategy.generate_signals(constant, strategy.indicators(constant, params),
                                    None, params).data
    assert (flat['signal'] == 0).all()


if __name__ == '__main__':
    _self_check()
