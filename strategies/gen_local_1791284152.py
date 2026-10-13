from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 10
    atr_bars: int = 14
    area_threshold: float = 0.20
    displacement_atr: float = 1.25
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1791284152'

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(params.path_bars + 2, params.atr_bars + 2))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.path_bars < 2 or params.atr_bars < 1 or params.hold_bars < 1:
            raise ValueError('Lookbacks and holding period must be positive; path_bars must be at least two.')
        if not all(np.isfinite(x) and x > 0 for x in (params.area_threshold, params.displacement_atr, params.stop_atr)):
            raise ValueError('Thresholds and stop multiplier must be finite and positive.')
        close = data['close'].astype(float)
        previous = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        scale = atr.where(atr > 0)
        length = params.path_bars
        start = close.shift(length)
        # Half of S(time, price) - S(price, time), using piecewise-linear interpolation.
        area = ((length + 1) * close / 2 + (length - 1) * start / 2
                - close.rolling(length, min_periods=length).sum()) / (length * scale)
        displacement = (close - start) / scale
        valid = np.isfinite(area) & np.isfinite(displacement) & np.isfinite(scale)
        long_candidate = valid & (area > params.area_threshold) & (displacement < -params.displacement_atr)
        short_candidate = valid & (area < -params.area_threshold) & (displacement > params.displacement_atr)
        confirmed_long = long_candidate & long_candidate.shift(1, fill_value=False)
        confirmed_short = short_candidate & short_candidate.shift(1, fill_value=False)
        confirmed = pd.Series(np.select([confirmed_long, confirmed_short], [1, -1], default=0), index=data.index)
        return pd.DataFrame({
            'true_range': true_range,
            'atr': atr,
            'signature_area': area,
            'displacement_atr': displacement,
            'long_candidate': long_candidate,
            'short_candidate': short_candidate,
            'confirmed_entry': confirmed,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['confirmed_entry'].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        frozen_atr = np.nan
        for i in range(len(data)):
            if position:
                # The previous close's entry order fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                invalid = not np.isfinite(entry_price) or entry_price <= 0
                stop_hit = np.isfinite(close[i]) and position * (close[i] - entry_price) < -params.stop_atr * frozen_atr
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid or stop_hit or expired:
                    position = 0
                    decisions[i] = 0
                    continue
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(close[i]):
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
                frozen_atr = atr[i]
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = SignatureParams(hold_bars=8, stop_atr=2.0)
    index = pd.date_range('2020-01-01', periods=6)
    for direction in (1, -1):
        data = pd.DataFrame({
            'open': 100.0,
            'high': 104.0,
            'low': 96.0,
            'close': [100.0, 100.0, 100.0, 100.0 - 3.0 * direction, 100.0, 100.0],
            'volume': 1000.0,
        }, index=index)
        indicators = pd.DataFrame({
            'atr': [1.0, 1.0, 100.0, 100.0, 100.0, 100.0],
            'confirmed_entry': [0, direction, 0, 0, 0, 0],
        }, index=index)
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
        assert result.data['signal'].tolist() == [0, 0, direction, direction, 0, 0]
        assert (result.data['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
