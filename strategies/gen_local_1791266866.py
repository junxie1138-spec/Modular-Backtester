from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DispersedLossParams:
    lookback: int = 12
    min_loss_days: int = 7
    max_concentration: float = 0.22
    min_stretch: float = 1.6
    atr_period: int = 14
    stop_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[DispersedLossParams]):
    strategy_id = 'gen_local_1791266866'

    @classmethod
    def params_type(cls) -> type[DispersedLossParams]:
        return DispersedLossParams

    @staticmethod
    def warmup_bars(params: DispersedLossParams) -> int:
        return int(max(200, params.lookback + 2, params.atr_period + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: DispersedLossParams) -> pd.DataFrame:
        if not (params.lookback >= 2 and params.atr_period >= 1
                and 1 <= params.min_loss_days <= params.lookback
                and 0 < params.max_concentration <= 1
                and params.min_stretch > 0 and params.stop_atr > 0
                and params.holding_bars >= 1):
            raise ValueError('Invalid strategy parameters')
        close = data['close']
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        previous_returns = returns.shift(1)
        losses = (-previous_returns).clip(lower=0)
        loss_mass = losses.rolling(params.lookback).sum()
        loss_squared_mass = losses.pow(2).rolling(params.lookback).sum()
        concentration = loss_squared_mass.div(loss_mass.pow(2).replace(0, np.nan))
        loss_days = previous_returns.lt(0).astype(float).where(previous_returns.notna()).rolling(params.lookback).sum()
        net_return = previous_returns.rolling(params.lookback).sum()
        volatility = previous_returns.rolling(params.lookback).std(ddof=0)
        stretch = (-net_return).div((volatility * np.sqrt(params.lookback)).replace(0, np.nan))
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_period).mean()
        ma200 = close.rolling(200).mean()
        result = pd.DataFrame({
            'return': returns,
            'loss_mass': loss_mass,
            'loss_squared_mass': loss_squared_mass,
            'loss_concentration': concentration,
            'loss_days': loss_days,
            'net_return': net_return,
            'return_volatility': volatility,
            'stretch': stretch,
            'true_range': true_range,
            'atr': atr,
            'ma200': ma200,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)
        result['entry'] = (
            result.notna().all(axis=1)
            & close.gt(ma200)
            & returns.gt(0)
            & loss_days.ge(params.min_loss_days)
            & concentration.le(params.max_concentration)
            & stretch.ge(params.min_stretch)
            & atr.gt(0)
        ).fillna(False)
        return result

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DispersedLossParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=np.int64)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        active = False
        pending = False
        stop_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_bar = i
                    # Entry is modeled at the next session's open; ATR is frozen at the decision close.
                    stop_price = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                stopped = np.isfinite(closes[i]) and closes[i] < stop_price
                expired = i - entry_bar + 1 >= params.holding_bars
                if stopped or expired:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and i + 1 < len(data):
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 96.0,
        'close': [100.0, 100.0, 99.0, 97.0, 100.0, 100.0, 100.0],
        'volume': 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        'entry': [True, False, False, False, False, False, False],
        'atr': [1.0, 1.0, 50.0, 50.0, 50.0, 50.0, 50.0],
    }, index=index)
    params = DispersedLossParams(stop_atr=2.0)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0]
    data['close'] = 100.0
    params.holding_bars = 3
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0]
    assert result['size'].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 200
    assert not GeneratedStrategy.indicators(data, params)['entry'].any()


if __name__ == '__main__':
    _self_check()
