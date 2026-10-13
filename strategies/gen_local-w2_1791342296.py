from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_window: int = 20
    vol_window: int = 20
    fast_vol_window: int = 5
    stretch_threshold: float = 1.5
    target_volatility: float = 0.10
    profit_target: float = 0.012
    max_hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1791342296'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.ma_window + 1, params.vol_window + 1, params.fast_vol_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.ma_window, params.vol_window, params.fast_vol_window) < 2:
            raise ValueError('Indicator windows must be at least two bars')
        if params.fast_vol_window >= params.vol_window:
            raise ValueError('Fast volatility window must be shorter than volatility window')
        if not all(np.isfinite(x) and x > 0 for x in (params.stretch_threshold, params.target_volatility, params.profit_target)):
            raise ValueError('Threshold, volatility target and profit target must be finite and positive')
        if not 3 <= params.max_hold_bars <= 5:
            raise ValueError('Holding period must be between three and five bars')
        close = data['close'].where(data['close'] > 0)
        returns = close.pct_change(fill_method=None)
        out = pd.DataFrame(index=data.index)
        out['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['price_std'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0).replace(0, np.nan)
        out['z'] = (close - out['ma']) / out['price_std']
        out['previous_z'] = out['z'].shift(1)
        out['return'] = returns
        out['regime_ma'] = close.rolling(200, min_periods=200).mean()
        out['vol'] = returns.rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0).replace(0, np.nan)
        out['fast_vol'] = returns.rolling(params.fast_vol_window, min_periods=params.fast_vol_window).std(ddof=0)
        out['annual_vol'] = out['vol'] * np.sqrt(252.0)
        out['exposure'] = (params.target_volatility / out['annual_vol']).clip(lower=0.01, upper=0.95)
        out['entry'] = (
            (close > out['regime_ma'])
            & (out['previous_z'] <= -params.stretch_threshold)
            & (out['z'] > out['previous_z'])
            & (out['return'] > 0)
            & (out['fast_vol'] < out['vol'])
            & out['exposure'].notna()
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.01, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        exposures = indicators['exposure'].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        position_size = 0.01
        for i in range(count):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                age = i - entry_bar + 1
                valid_price = np.isfinite(entry_price) and entry_price > 0
                target_hit = valid_price and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if not valid_price or target_hit or age >= params.max_hold_bars:
                    active = False
                else:
                    signals[i] = 1
                sizes[i] = position_size
            elif entries[i] and np.isfinite(exposures[i]) and exposures[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                position_size = float(np.clip(exposures[i], 0.01, 0.95))
                signals[i] = 1
                sizes[i] = position_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry': [True, False, False, False, False, False, False, False], 'exposure': 0.4}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[2], 'close'] = 102.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert result['size'].gt(0).all()
    assert result.loc[index[1], 'size'] == 0.4
    assert strategy.warmup_bars(params) >= 200
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived['entry'].any()


if __name__ == '__main__':
    _self_check()
