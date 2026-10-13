from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MemoryDebtParams:
    frac_order: float = 0.45
    frac_window: int = 32
    debt_window: int = 3
    volatility_window: int = 20
    atr_window: int = 14
    repayment_ratio: float = 0.65
    target_volatility: float = 0.10
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[MemoryDebtParams]):
    strategy_id = 'gen_local_1790864011'

    @classmethod
    def params_type(cls):
        return MemoryDebtParams

    def warmup_bars(self, params: MemoryDebtParams) -> int:
        return int(max(200, params.frac_window + params.debt_window + 1,
                       params.volatility_window + params.debt_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: MemoryDebtParams) -> pd.DataFrame:
        if (params.frac_window < 2 or params.debt_window < 1
                or params.volatility_window < 2 or params.atr_window < 1
                or not 0.0 < params.frac_order < 1.0
                or not 0.0 < params.repayment_ratio <= 1.0
                or params.target_volatility <= 0.0
                or params.breakeven_pct <= 0.0 or params.trail_atr <= 0.0
                or not 3 <= params.holding_bars <= 5):
            raise ValueError('Invalid strategy parameters')
        close = data['close'].astype(float)
        out = pd.DataFrame(index=data.index)
        weights = np.ones(params.frac_window, dtype=float)
        for lag in range(1, params.frac_window):
            weights[lag] = -weights[lag - 1] * (params.frac_order - lag + 1) / lag
        # Finite, causal fractional differencing; retain every kernel term.
        fractional_close = close.copy()
        for lag in range(1, params.frac_window):
            fractional_close = fractional_close + weights[lag] * close.shift(lag)
        out['fractional_close'] = fractional_close
        out['fractional_impulse'] = fractional_close.diff()
        impulse = out['fractional_impulse']
        out['negative_debt'] = (-impulse.clip(upper=0.0)).shift(1).rolling(
            params.debt_window, min_periods=params.debt_window).sum()
        out['negative_count'] = impulse.lt(0.0).astype(float).where(
            impulse.notna()).shift(1).rolling(
            params.debt_window, min_periods=params.debt_window).sum()
        out['return'] = close.pct_change(fill_method=None)
        out['realized_volatility'] = out['return'].rolling(
            params.volatility_window, min_periods=params.volatility_window).std(ddof=0) * np.sqrt(252.0)
        out['prior_volatility'] = out['realized_volatility'].shift(params.debt_window)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out['entry_atr'] = out['atr'].shift(1)
        out['target_size'] = (params.target_volatility / out['realized_volatility'].replace(
            0.0, np.nan)).clip(lower=0.01, upper=0.95)
        out['entry'] = (
            close.gt(out['ma200'])
            & out['negative_count'].eq(params.debt_window)
            & out['negative_debt'].gt(0.0)
            & impulse.gt(params.repayment_ratio * out['negative_debt'])
            & out['realized_volatility'].gt(0.0)
            & out['realized_volatility'].lt(out['prior_volatility'])
            & out['atr'].gt(0.0)
            & out['target_size'].notna()
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: MemoryDebtParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.95, dtype=float)
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entry_atr = indicators['entry_atr'].to_numpy(dtype=float)
        regime = indicators['ma200'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        targets = indicators['target_size'].to_numpy(dtype=float)
        active = False
        armed = False
        entry_bar = -1
        entry_price = peak = stop = np.nan
        exposure = 0.95
        for i in range(n):
            if active:
                # A preceding close decision enters at this bar's open.
                if i == entry_bar:
                    entry_price = opening[i]
                    peak = entry_price
                    stop = entry_price - params.trail_atr * entry_atr[i]
                valid = (np.isfinite(entry_price) and entry_price > 0.0
                         and np.isfinite(stop) and np.isfinite(close[i])
                         and np.isfinite(atr[i]) and atr[i] > 0.0
                         and np.isfinite(regime[i]))
                if not valid:
                    active = False
                    continue
                # Stops are evaluated at closes and executed next open.
                peak = max(peak, close[i])
                if close[i] >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, peak - params.trail_atr * atr[i])
                if (close[i] <= stop or close[i] <= regime[i]
                        or i - entry_bar + 1 >= params.holding_bars):
                    active = False
                    continue
                signals[i] = 1
                sizes[i] = exposure
            elif entries[i] and np.isfinite(targets[i]) and targets[i] > 0.0:
                active = True
                armed = False
                entry_bar = i + 1
                exposure = float(targets[i])
                signals[i] = 1
                sizes[i] = exposure
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': 100.0, 'high': 104.0, 'low': 99.0,
        'close': [100.0, 102.0, 103.0, 101.0, 100.0, 100.0],
        'volume': 1000000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': 1.0, 'entry_atr': 1.0, 'ma200': 99.0,
        'entry': [True, False, False, False, False, False],
        'target_size': 0.5,
    }, index=index)
    strategy = GeneratedStrategy()
    params = MemoryDebtParams()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    assert result.loc[index[1]:index[3], 'size'].eq(0.5).all()
    assert result['size'].gt(0.0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
