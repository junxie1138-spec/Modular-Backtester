from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureRecoveryParams:
    path_window: int = 6
    area_threshold: float = 0.08
    volume_threshold: float = 0.10
    vol_window: int = 20
    target_vol: float = 0.10
    min_size: float = 0.10
    max_size: float = 0.95
    profit_target: float = 0.018
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[SignatureRecoveryParams]):
    strategy_id = 'gen_local-w2_1791295311'

    @classmethod
    def params_type(cls) -> type[SignatureRecoveryParams]:
        return SignatureRecoveryParams

    def warmup_bars(self, params: SignatureRecoveryParams) -> int:
        return int(max(params.path_window, params.vol_window) + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureRecoveryParams) -> pd.DataFrame:
        if params.path_window < 2 or params.vol_window < 2:
            raise ValueError('Lookback windows must be at least two.')
        if not 0 < params.area_threshold < 0.25:
            raise ValueError('area_threshold must lie between zero and 0.25.')
        if not 0 < params.volume_threshold < 1:
            raise ValueError('volume_threshold must lie between zero and one.')
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError('Size bounds must satisfy 0 < min_size <= max_size <= 1.')
        if params.target_vol <= 0 or params.profit_target <= 0 or not 3 <= params.hold_bars <= 5:
            raise ValueError('Positive volatility/profit targets and a three-to-five-bar holding limit are required.')

        close = data['close'].where(data['close'] > 0)
        returns = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        absolute_returns = returns.abs()
        volume = data['volume'].where(data['volume'] >= 0).replace([np.inf, -np.inf], np.nan)
        signed_volume = np.sign(returns) * volume
        w = params.path_window
        x = pd.Series(0.0, index=data.index)
        y = pd.Series(0.0, index=data.index)
        area = pd.Series(0.0, index=data.index)
        # Second-order antisymmetric signature of (cumulative return, total variation).
        # Earlier selling followed by buying gives negative signed area.
        for lag in range(w - 1, -1, -1):
            dx = returns.shift(lag)
            dy = absolute_returns.shift(lag)
            area = area + 0.5 * (x * dy - y * dx)
            x = x + dx
            y = y + dy

        rolling_volume = volume.rolling(w, min_periods=w).sum()
        rolling_signed_volume = signed_volume.rolling(w, min_periods=w).sum()
        normalized_area = area / y.pow(2).replace(0.0, np.nan)
        turnover = rolling_signed_volume / rolling_volume.replace(0.0, np.nan)
        annual_vol = returns.rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        target_size = (params.target_vol / annual_vol.replace(0.0, np.nan)).clip(params.min_size, params.max_size)
        return pd.DataFrame({
            'log_return': returns,
            'absolute_return': absolute_returns,
            'signed_volume': signed_volume,
            'path_displacement': x,
            'path_variation': y,
            'signature_area': area,
            'normalized_area': normalized_area,
            'rolling_volume': rolling_volume,
            'rolling_signed_volume': rolling_signed_volume,
            'turnover': turnover,
            'annual_vol': annual_vol,
            'target_size': target_size,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureRecoveryParams) -> SignalFrame:
        eligible = (
            (indicators['normalized_area'] <= -params.area_threshold)
            & (indicators['turnover'] >= params.volume_threshold)
            & (indicators['annual_vol'] > 0)
            & np.isfinite(indicators['target_size'])
        ).fillna(False).to_numpy(dtype=bool)
        desired = indicators['target_size'].fillna(params.min_size).to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), params.min_size, dtype=float)
        entry_bar = -1
        entry_price = np.nan
        frozen_size = params.min_size

        for i in range(len(data)):
            if entry_bar >= 0:
                # Observe the entry open only when its bar has actually arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    entry_bar = -1
                    continue
                expired = i - entry_bar + 1 >= params.hold_bars
                # Close-confirmed target: flatten at the following open.
                reached = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                size[i] = frozen_size
                if expired or reached:
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif eligible[i] and np.isfinite(closes[i]) and closes[i] > 0:
                entry_bar = i + 1
                frozen_size = float(desired[i])
                signal[i] = 1
                size[i] = frozen_size

        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = SignatureRecoveryParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'normalized_area': [-0.2] + [0.0] * 6, 'turnover': 0.5, 'annual_vol': 0.2, 'target_size': 0.5}, index=index)
    timed = strategy.generate_signals(data, ind, None, params).data
    assert timed['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert timed.loc[index[1], 'size'] == 0.5
    data.loc[index[2], 'close'] = 103.0
    targeted = strategy.generate_signals(data, ind, None, params).data
    assert targeted['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0]
    ind['turnover'] = -0.5
    assert strategy.generate_signals(data, ind, None, params).data['signal'].eq(0).all()
    ind['turnover'] = 0.5
    ind['normalized_area'] = 0.2
    assert strategy.generate_signals(data, ind, None, params).data['signal'].eq(0).all()
    warm = strategy.indicators(data, params)
    assert strategy.generate_signals(data, warm, None, params).data['signal'].eq(0).all()
    assert strategy.warmup_bars(params) == 21


if __name__ == '__main__':
    _self_check()
