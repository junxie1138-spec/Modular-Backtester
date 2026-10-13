from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RotationParams:
    seasonal_window: int = 252
    entry_z: float = 1.25
    atr_period: int = 14
    trail_atr: float = 1.5
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[RotationParams]):
    strategy_id = "gen_local-w2_1791217278"

    @classmethod
    def params_type(cls) -> type[RotationParams]:
        return RotationParams

    def warmup_bars(self, params: RotationParams) -> int:
        return int(max(200, max(2, params.atr_period) + 1,
                       max(30, params.seasonal_window) + 4))

    def indicators(self, data: pd.DataFrame, params: RotationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous = data['close'].shift(1).replace(0.0, np.nan)
        opening = data['open'].replace(0.0, np.nan)
        out['gap'] = (data['open'] / previous - 1.0).replace([np.inf, -np.inf], np.nan)
        out['intraday'] = (data['close'] / opening - 1.0).replace([np.inf, -np.inf], np.nan)
        x, y = out['gap'], out['intraday']
        out['area'] = 0.5 * (
            x.shift(2) * y.shift(1) - y.shift(2) * x.shift(1)
            + x.shift(2) * y - y.shift(2) * x
            + x.shift(1) * y - y.shift(1) * x
        )
        out['weekday'] = data.index.dayofweek
        out['seasonal_mean'] = np.nan
        out['seasonal_std'] = np.nan
        window = max(30, int(params.seasonal_window))
        minimum = max(4, window // 20)
        for day in range(5):
            mask = out['weekday'].eq(day)
            history = out['area'].where(mask).shift(1)
            mean = history.rolling(window, min_periods=minimum).mean()
            std = history.rolling(window, min_periods=minimum).std()
            out.loc[mask, 'seasonal_mean'] = mean.loc[mask]
            out.loc[mask, 'seasonal_std'] = std.loc[mask]
        out['rotation_z'] = (
            (out['area'] - out['seasonal_mean'])
            / out['seasonal_std'].where(out['seasonal_std'] > 1e-12)
        ).replace([np.inf, -np.inf], np.nan)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        period = max(2, int(params.atr_period))
        out['atr'] = out['true_range'].rolling(period, min_periods=period).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RotationParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ma = indicators['ma200'].to_numpy(dtype=float)
        z = indicators['rotation_z'].to_numpy(dtype=float)
        threshold = max(0.01, float(params.entry_z))
        candidate = np.where((z >= threshold) & (close > ma), 1,
                             np.where((z <= -threshold) & (close < ma), -1, 0))
        valid = np.isfinite(close) & np.isfinite(atr) & (atr > 0) & np.isfinite(ma) & np.isfinite(z)
        candidate = np.where(valid, candidate, 0)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        highest = np.nan
        stop = -np.inf
        hold = min(2, max(1, int(params.hold_days)))
        multiple = max(0.01, float(params.trail_atr))
        for i in range(len(data)):
            if position:
                age += 1
                if not valid[i]:
                    position = 0
                else:
                    highest = close[i] if not np.isfinite(highest) else max(highest, close[i])
                    stop = max(stop, highest - multiple * atr[i])
                    if close[i] <= stop or age >= hold:
                        position = 0
                decisions[i] = position
                continue
            if candidate[i]:
                position = int(candidate[i])
                age = 0
                highest = np.nan
                stop = -np.inf
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = RotationParams()
    index = pd.bdate_range('2020-01-01', periods=600)
    rng = np.random.default_rng(19)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(index))))
    opening = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0.0, 0.004, len(index)))
    data = pd.DataFrame({'open': opening, 'high': np.maximum(opening, close) * 1.005,
                         'low': np.minimum(opening, close) * 0.995,
                         'close': close, 'volume': 1000000.0}, index=index)
    indicator = GeneratedStrategy.indicators(None, data, params)
    result = GeneratedStrategy.generate_signals(None, data, indicator, None, params).data
    prefix = data.iloc[:450]
    prefix_indicator = GeneratedStrategy.indicators(None, prefix, params)
    prefix_result = GeneratedStrategy.generate_signals(None, prefix, prefix_indicator, None, params).data
    assert result.iloc[:450].equals(prefix_result)
    assert result['signal'].isin([-1, 0, 1]).all()
    assert (result['size'] > 0).all()
    assert result['signal'].iloc[0] == 0
    assert GeneratedStrategy.warmup_bars(None, params) >= params.seasonal_window + 4
    controlled = pd.DataFrame({'atr': 1.0, 'ma200': 90.0, 'rotation_z': 2.0}, index=index[:5])
    probe = data.iloc[:5].copy()
    probe['close'] = [100.0, 103.0, 100.0, 100.0, 100.0]
    stopped = GeneratedStrategy.generate_signals(None, probe, controlled, None, params).data
    assert stopped['signal'].tolist() == [0, 1, 1, 0, 1]


if __name__ == '__main__':
    _self_check()
