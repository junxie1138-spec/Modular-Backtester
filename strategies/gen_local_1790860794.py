from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    normalization_window: int = 30
    area_window: int = 5
    baseline_window: int = 60
    changepoint_threshold: float = 2.0
    profit_target: float = 0.025
    holding_bars: int = 10
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1790860794'

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(params.normalization_window + params.area_window + params.baseline_window + 2)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if min(params.normalization_window, params.area_window, params.baseline_window) < 2:
            raise ValueError('Indicator windows must be at least two bars.')
        if params.changepoint_threshold <= 0:
            raise ValueError('Changepoint threshold must be positive.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        returns = close.pct_change(fill_method=None)
        volatility = returns.rolling(params.normalization_window).std(ddof=0).shift(1)
        out['return'] = returns
        out['volatility'] = volatility
        out['return_step'] = returns.div(volatility.where(volatility > 1e-12)).clip(-5.0, 5.0)
        log_volume = np.log1p(data['volume'].where(data['volume'] >= 0))
        volume_mean = log_volume.rolling(params.normalization_window).mean().shift(1)
        volume_std = log_volume.rolling(params.normalization_window).std(ddof=0).shift(1)
        out['log_volume'] = log_volume
        out['volume_mean'] = volume_mean
        out['volume_std'] = volume_std
        out['volume_step'] = log_volume.sub(volume_mean).div(volume_std.where(volume_std > 1e-12)).clip(-5.0, 5.0)
        x = out['return_step']
        y = out['volume_step']
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, params.area_window):
            pair = x.shift(lag) * y - y.shift(lag) * x
            area = area + pair.rolling(params.area_window - lag).sum()
        out['area'] = area / (params.area_window * (params.area_window - 1))
        out['area_mean'] = out['area'].rolling(params.baseline_window).mean().shift(1)
        out['area_std'] = out['area'].rolling(params.baseline_window).std(ddof=0).shift(1)
        out['changepoint'] = out['area'].sub(out['area_mean']).div(out['area_std'].where(out['area_std'] > 1e-12))
        out['momentum'] = returns.rolling(params.area_window).sum()
        z = out['changepoint']
        ready = z.notna() & z.shift(1).notna() & out['momentum'].notna()
        crossing = (z.abs() >= params.changepoint_threshold) & (z.shift(1).abs() < params.changepoint_threshold)
        aligned = z * out['momentum'] > 0
        out['candidate'] = np.where(ready & crossing & aligned, np.sign(out['momentum']), 0).astype(int)
        out['conviction'] = (z.abs() / (2.0 * params.changepoint_threshold)).clip(0.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        if params.holding_bars < 1 or params.profit_target <= 0:
            raise ValueError('Holding period and profit target must be positive.')
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1.')
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        candidates = indicators['candidate'].to_numpy(dtype=int)
        conviction = indicators['conviction'].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        position_size = 1.0
        for i in range(n):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                        continue
                gain = position * (closes[i] / entry_price - 1.0)
                age = i - entry_bar + 1
                if gain >= params.profit_target or age >= params.holding_bars:
                    position = 0
                    continue
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
                strength = float(np.clip(conviction[i], 0.0, 1.0))
                position_size = params.min_size + (params.max_size - params.min_size) * strength
            signals[i] = position
            sizes[i] = position_size if position else 1.0
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=11)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    data.loc[index[1], 'close'] = 103.0
    data.loc[index[8], 'close'] = 97.0
    indicators = pd.DataFrame({'candidate': [1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0], 'conviction': 0.8}, index=index)
    strategy = GeneratedStrategy()
    params = SignatureParams(profit_target=0.02, holding_bars=3)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, -1, -1, -1, 0, 0, -1, 0, 0]
    assert result['size'].gt(0).all()
    expected_size = params.min_size + (params.max_size - params.min_size) * 0.8
    assert np.allclose(result.loc[result['signal'] != 0, 'size'], expected_size)
