from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    trend_window: int = 8
    correlation_window: int = 60
    echo_threshold: float = 0.35
    min_correlation: float = 0.15
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 3.0
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = 'gen_local-w2_1791290982'

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return int(max(200, params.trend_window + params.correlation_window + 12, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if min(params.trend_window, params.correlation_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two.')
        if params.echo_threshold <= 0 or not 0 < params.min_correlation <= 2:
            raise ValueError('Invalid echo thresholds.')
        if params.breakeven_pct <= 0 or params.trail_atr <= 0 or params.max_hold < 1:
            raise ValueError('Invalid exit parameters.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['log_return'] = np.log(close.where(close > 0)).diff()
        out['slope'] = out['log_return'].rolling(params.trend_window, min_periods=params.trend_window).mean()
        out['innovation'] = out['slope'].diff()
        out['lag5'] = out['innovation'].shift(5)
        out['lag10'] = out['innovation'].shift(10)
        window = params.correlation_window
        out['rho5'] = out['innovation'].rolling(window, min_periods=window).corr(out['lag5']).shift(1).clip(0.0, 1.0)
        out['rho10'] = out['innovation'].rolling(window, min_periods=window).corr(out['lag10']).shift(1).clip(0.0, 1.0)
        out['noise'] = out['innovation'].rolling(window, min_periods=window).std(ddof=0).shift(1)
        out['persistence'] = out['rho5'] + out['rho10']
        out['echo'] = out['rho5'] * out['lag5'] + out['rho10'] * out['lag10']
        out['echo_score'] = out['echo'] / out['noise'].where(out['noise'] > 0)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[['ma200', 'slope', 'innovation', 'echo_score', 'persistence', 'atr']].notna().all(axis=1)
        ready &= (out['atr'] > 0) & (out['persistence'] >= params.min_correlation)
        long_entry = ready & (close > out['ma200']) & (out['slope'] > 0) & (out['innovation'] > 0) & (out['echo_score'] >= params.echo_threshold)
        short_entry = ready & (close < out['ma200']) & (out['slope'] < 0) & (out['innovation'] < 0) & (out['echo_score'] <= -params.echo_threshold)
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        count = len(data)
        raw = np.zeros(count, dtype=int)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        position = 0
        entry_price = 0.0
        best = 0.0
        stop = 0.0
        armed = False
        held = 0
        for i in range(count):
            # A decision from yesterday establishes today's entry at its open.
            if position == 0 and i > 0 and raw[i - 1] != 0:
                if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                position = int(raw[i - 1])
                entry_price = float(opens[i])
                best = position * entry_price
                stop = best - params.trail_atr * atr[i - 1]
                armed = False
                held = 0
            if position != 0:
                held += 1
                if not (np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0):
                    position = 0
                    continue
                signed_close = position * closes[i]
                # Stops are close-confirmed; the shifted zero exits next open.
                if signed_close <= stop or held >= params.max_hold:
                    position = 0
                    continue
                best = max(best, signed_close)
                if best - position * entry_price >= entry_price * params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed-price stop only rises; a short's actual stop falls.
                    stop = max(stop, position * entry_price, best - params.trail_atr * atr[i])
                raw[i] = position
            elif i + 1 < count:
                raw[i] = entries[i]
        df = pd.DataFrame({'signal': raw, 'size': np.ones(count, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = EchoParams(breakeven_pct=0.02)
    for side in (1, -1):
        prices = 100.0 + side * np.array([0, 0, 3, 5, 3, 2, 2, 2], dtype=float)
        index = pd.date_range('2020-01-01', periods=len(prices))
        data = pd.DataFrame({'open': prices, 'high': prices + 1, 'low': prices - 1, 'close': prices, 'volume': 1000.0}, index=index)
        derived = pd.DataFrame({'entry': [side, 0, 0, 0, 0, 0, 0, 0], 'atr': 1.0}, index=index)
        result = strategy.generate_signals(data, derived, None, params).data
        assert result['signal'].tolist() == [0, side, side, side, side, side, 0, 0]
        assert (result['size'] > 0).all()
        actual = strategy.indicators(data, params)
        assert actual['entry'].eq(0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
