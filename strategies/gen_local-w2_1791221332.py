from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    depth_lookback: int = 20
    autocorr_window: int = 40
    min_depth: float = 0.015
    min_correlation: float = 0.15
    atr_period: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = 'gen_local-w2_1791221332'

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return int(max(200, params.depth_lookback + params.autocorr_window + 1,
                       params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if (params.depth_lookback < 2 or params.autocorr_window < 3
                or params.atr_period < 2 or not 3 <= params.hold_days <= 5
                or not 0 < params.min_depth < 1
                or not 0 < params.min_correlation < 1
                or not np.isfinite(params.breakeven_pct)
                or params.breakeven_pct <= 0
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0):
            raise ValueError('Invalid strategy parameters')
        close = data['close'].where(data['close'] > 0)
        out = pd.DataFrame(index=data.index)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['peak'] = close.rolling(params.depth_lookback,
                                    min_periods=params.depth_lookback).max()
        out['depth'] = 1.0 - close / out['peak']
        out['depth_change'] = out['depth'].diff()
        window = params.autocorr_window
        changes = out['depth_change']
        out['mean_change'] = changes.rolling(window, min_periods=window).mean()
        out['rho'] = changes.rolling(window, min_periods=window).corr(changes.shift(1))
        out['rho'] = out['rho'].replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        out['forecast'] = out['mean_change'] + out['rho'] * (changes - out['mean_change'])
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_period,
                                              min_periods=params.atr_period).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        close = data['close']
        required = ['ma200', 'depth', 'depth_change', 'rho', 'forecast', 'atr']
        ready = pd.Series(np.isfinite(indicators[required].to_numpy()).all(axis=1),
                          index=data.index)
        ready &= np.isfinite(close) & (close > 0) & (indicators['atr'] > 0)
        deepening = ready & (indicators['depth'] >= params.min_depth)
        deepening &= indicators['depth_change'] > 0
        longs = (deepening & (close > indicators['ma200'])
                 & (indicators['rho'] <= -params.min_correlation)
                 & (indicators['forecast'] < 0))
        shorts = (deepening & (close < indicators['ma200'])
                  & (indicators['rho'] >= params.min_correlation)
                  & (indicators['forecast'] > 0))
        opportunities = np.select([longs.fillna(False), shorts.fillna(False)],
                                  [1, -1], default=0)
        opens = data['open'].to_numpy(dtype=float)
        closes = close.to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry = stop = best = np.nan
        armed = False

        for i in range(len(data)):
            if side:
                # Yesterday's decision fills at today's observed open.
                if i == entry_bar:
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                    else:
                        # Direction-adjusted stops ratchet upward for both sides;
                        # the corresponding short stop price ratchets downward.
                        stop = side * entry - params.trail_atr * atr[i - 1]
                        best = side * entry
                        armed = False
                if side:
                    signed_close = side * closes[i]
                    if (not np.isfinite(signed_close) or closes[i] <= 0
                            or not np.isfinite(atr[i]) or atr[i] <= 0):
                        side = 0
                    elif (signed_close <= stop
                          or i - entry_bar + 1 >= params.hold_days):
                        side = 0
                    else:
                        best = max(best, signed_close)
                        if best - side * entry >= entry * params.breakeven_pct:
                            armed = True
                        if armed:
                            stop = max(stop, side * entry,
                                       best - params.trail_atr * atr[i])
            elif opportunities[i]:
                side = int(opportunities[i])
                entry_bar = i + 1
            decisions[i] = side

        # Stops are evaluated at closes; exits execute at the next bar's open.
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range('2020-01-01', periods=7)
    for side in (1, -1):
        for trailing in (True, False):
            params = DrawdownParams(hold_days=5 if trailing else 4)
            moves = np.array([0, 0, 2, 1, 0.25, 0.25, 0.25]) if trailing else np.zeros(7)
            close = 100.0 + side * moves
            data = pd.DataFrame({'open': 100.0, 'high': np.maximum(close, 100.0) + 1,
                                 'low': np.minimum(close, 100.0) - 1,
                                 'close': close, 'volume': 1000.0}, index=index)
            ind = pd.DataFrame({'ma200': 100.0 - side * 10, 'depth': 0.03,
                                'depth_change': 0.001, 'rho': -side * 0.5,
                                'forecast': -side * 0.001, 'atr': 1.0}, index=index)
            ind.loc[index[1:], 'depth_change'] = 0.0
            result = strategy.generate_signals(data, ind, None, params).data
            assert result['signal'].tolist() == [0, side, side, side, side, 0, 0]
            assert (result['size'] > 0).all()
    assert strategy.warmup_bars(DrawdownParams(depth_lookback=180, autocorr_window=40)) == 221


if __name__ == '__main__':
    _self_check()
