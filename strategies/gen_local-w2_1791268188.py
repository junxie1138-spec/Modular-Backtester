from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapReleaseParams:
    gap_window: int = 4
    min_quiet_gaps: int = 2
    down_gap_floor: float = 0.001
    volume_window: int = 20
    release_volume: float = 1.25
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapReleaseParams]):
    strategy_id = 'gen_local-w2_1791268188'

    @classmethod
    def params_type(cls) -> type[GapReleaseParams]:
        return GapReleaseParams

    def warmup_bars(self, params: GapReleaseParams) -> int:
        return int(max(200, params.volume_window + params.gap_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapReleaseParams) -> pd.DataFrame:
        if not (params.volume_window >= 1 and params.gap_window >= 1
                and 1 <= params.min_quiet_gaps <= params.gap_window
                and params.down_gap_floor > 0 and params.release_volume > 0):
            raise ValueError('Invalid gap or volume parameters')
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1).replace(0.0, np.nan)
        out['gap'] = data['open'] / previous_close - 1.0
        out['volume_baseline'] = data['volume'].shift(1).rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean()
        out['relative_volume'] = data['volume'] / out['volume_baseline'].replace(0.0, np.nan)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['gap_ceiling'] = data['high'].shift(1).rolling(
            params.gap_window, min_periods=params.gap_window
        ).max()
        out['quiet_down_gap'] = (
            out['gap'].le(-params.down_gap_floor)
            & out['relative_volume'].gt(0.0)
            & out['relative_volume'].le(1.0)
        ).fillna(False)
        out['quiet_gap_count'] = out['quiet_down_gap'].astype(float).shift(1).rolling(
            params.gap_window, min_periods=params.gap_window
        ).sum()
        out['entry'] = (
            out['quiet_gap_count'].ge(params.min_quiet_gaps)
            & data['open'].gt(out['gap_ceiling'])
            & data['close'].gt(data['open'])
            & out['relative_volume'].ge(params.release_volume)
            & data['close'].gt(out['ma200'])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapReleaseParams,
    ) -> SignalFrame:
        if params.profit_target <= 0 or params.max_hold_bars < 1:
            raise ValueError('Profit target and holding period must be positive')
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_price = 0.0
        entry_bar = -1
        for i in range(len(data)):
            if active:
                # Yesterday's entry decision fills at today's open.
                if entry_bar < 0:
                    entry_price = opens[i]
                    entry_bar = i
                if not np.isfinite(entry_price) or entry_price <= 0:
                    active = False
                elif (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                ) or i - entry_bar + 1 >= params.max_hold_bars:
                    # Close-confirmed exits execute at the following open.
                    active = False
                raw[i] = int(active)
            elif entries[i]:
                active = True
                entry_bar = -1
                raw[i] = 1
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({
        'open': [100.0] * 5,
        'high': [104.0] * 5,
        'low': [99.0] * 5,
        'close': [100.0, 100.0, 103.0, 100.0, 100.0],
        'volume': [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({'entry': [True, False, False, False, False]}, index=index)
    params = GapReleaseParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    assert result['size'].gt(0).all()
    data['close'] = 100.0
    params.max_hold_bars = 2
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    warmup = GeneratedStrategy.indicators(None, data, params)
    assert not warmup['entry'].any()
    empty = GeneratedStrategy.generate_signals(
        None, data.iloc[:0], indicators.iloc[:0], None, params
    ).data
    assert empty.empty


if __name__ == '__main__':
    _self_check()
