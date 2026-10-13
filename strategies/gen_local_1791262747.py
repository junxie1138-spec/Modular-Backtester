from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 126
    percentile_window: int = 252
    depth_quantile: float = 0.90
    volume_window: int = 20
    volume_quantile: float = 0.90
    reclaim_window: int = 8
    atr_window: int = 14
    stop_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791262747'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(
            params.peak_window + params.percentile_window + params.reclaim_window + 1,
            params.volume_window + params.reclaim_window + 1,
            params.atr_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['depth'] = 1.0 - close / out['peak'].where(out['peak'] > 0)
        out['depth_threshold'] = out['depth'].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.depth_quantile)
        out['volume_threshold'] = data['volume'].shift(1).rolling(
            params.volume_window, min_periods=params.volume_window
        ).quantile(params.volume_quantile)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out['deep'] = (
            (out['depth'] > 0)
            & (out['depth'] >= out['depth_threshold'])
        ).fillna(False)
        out['midpoint'] = (data['open'] + close) / 2.0
        out['panic'] = (
            out['deep']
            & (close < data['open'])
            & (data['volume'] > out['volume_threshold'])
            & (out['atr'] > 0)
        ).fillna(False)
        positions = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        out['anchor'] = out['midpoint'].where(out['panic']).ffill().shift(1)
        out['anchor_id'] = positions.where(out['panic']).ffill().shift(1)
        out['anchor_age'] = positions - out['anchor_id']
        out['entry'] = (
            out['deep']
            & out['anchor_age'].between(1, params.reclaim_window)
            & (close > out['anchor'])
            & (previous_close <= out['anchor'])
            & (out['atr'] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        anchor_ids = indicators['anchor_id'].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        stop = np.nan
        consumed_anchor = -1.0
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Fix the stop to the fill-bar open and the decision-bar ATR.
                    stop = opening[i] - params.stop_atr * atr[i - 1]
                    if not np.isfinite(opening[i]) or opening[i] <= 0:
                        active = False
                if active and (
                    not np.isfinite(close[i])
                    or not np.isfinite(stop)
                    or close[i] < stop
                    or i - entry_bar + 1 >= params.holding_bars
                ):
                    active = False
            elif (
                entries[i]
                and np.isfinite(close[i])
                and close[i] > 0
                and np.isfinite(atr[i])
                and atr[i] > 0
                and np.isfinite(anchor_ids[i])
                and anchor_ids[i] != consumed_anchor
            ):
                active = True
                entry_bar = i + 1
                consumed_anchor = anchor_ids[i]
            decisions[i] = int(active)
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': [100.0, 105.0, 104.0, 101.0, 99.0, 99.0, 99.0],
        'close': [100.0, 104.0, 101.0, 99.0, 99.0, 99.0, 99.0],
    }, index=index)
    derived = pd.DataFrame({
        'atr': [2.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        'entry': [True, False, False, False, False, False, False],
        'anchor_id': [0.0] * 7,
    }, index=index)
    stopped = GeneratedStrategy.generate_signals(data, derived, None, Params()).data
    assert stopped['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0]
    data['close'] = 104.0
    expired = GeneratedStrategy.generate_signals(
        data, derived, None, Params(holding_bars=3)
    ).data
    assert expired['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0]
    assert (expired['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
