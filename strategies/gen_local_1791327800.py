from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 4
    rank_bars: int = 126
    atr_bars: int = 14
    extreme_quantile: float = 0.85
    breakeven_pct: float = 0.006
    trail_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1791327800'

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(params.atr_bars + params.path_bars + params.rank_bars + 3)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.path_bars < 2 or params.rank_bars < 2 or params.atr_bars < 1:
            raise ValueError('Lookbacks must be positive; path and rank windows require at least two bars.')
        if not 0.0 < params.extreme_quantile < 1.0:
            raise ValueError('extreme_quantile must be between zero and one.')
        if params.breakeven_pct <= 0.0 or params.trail_atr <= 0.0:
            raise ValueError('Exit parameters must be positive.')
        previous_close = data['close'].shift(1)
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        scale = atr.shift(1).where(atr.shift(1) > 0.0)
        x = data['close'].diff() / scale
        y = tr.diff() / scale
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, params.path_bars):
            wedge = x.shift(lag) * y - y.shift(lag) * x
            window = params.path_bars - lag
            area = area + 0.5 * wedge.rolling(window, min_periods=window).sum()
        innovation = area.diff()
        threshold = innovation.abs().shift(1).rolling(
            params.rank_bars, min_periods=params.rank_bars
        ).quantile(params.extreme_quantile)
        rotation = (area * area.shift(1) < 0.0) & (innovation.abs() > threshold)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(threshold) & np.isfinite(area)
        entry = pd.Series(0, index=data.index, dtype=int)
        entry.loc[valid & rotation & (x > 0.0)] = 1
        entry.loc[valid & rotation & (x < 0.0)] = -1
        return pd.DataFrame({
            'true_range': tr,
            'atr': atr,
            'price_increment': x,
            'range_increment': y,
            'signature_area': area,
            'area_innovation': innovation,
            'innovation_threshold': threshold,
            'entry': entry,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        side = 0
        entry = np.nan
        stop = -np.inf
        armed = False
        held = 0
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            if not np.isfinite(prices[i]).all() or min(prices[i]) <= 0.0:
                side = 0
                entry = np.nan
                continue
            if side:
                if np.isnan(entry):
                    if i == 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0.0:
                        side = 0
                        continue
                    # The preceding close's decision fills at this bar's open.
                    entry = opening
                    stop = side * entry - params.trail_atr * atr[i - 1]
                    armed = False
                    held = 0
                held += 1
                # Signed prices make the stop ratchet upward for either direction.
                favorable = high if side == 1 else -low
                adverse = low if side == 1 else -high
                signed_close = side * close
                exit_now = adverse <= stop
                if not exit_now:
                    if favorable - side * entry >= entry * params.breakeven_pct:
                        armed = True
                    if armed:
                        stop = max(stop, side * entry)
                        if np.isfinite(atr[i]) and atr[i] > 0.0:
                            stop = max(stop, favorable - params.trail_atr * atr[i])
                    exit_now = signed_close <= stop
                # Stop detection requests an exit at the following open.
                if exit_now or held >= 2:
                    side = 0
                    entry = np.nan
                raw[i] = side
                continue
            if entries[i] in (-1, 1) and np.isfinite(atr[i]) and atr[i] > 0.0:
                side = int(entries[i])
                entry = np.nan
                raw[i] = side
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = SignatureParams()
    index = pd.date_range('2020-01-01', periods=4)
    for side in (1, -1):
        # First held session reaches the activation threshold and retraces.
        close = 100.5 if side == 1 else 99.5
        data = pd.DataFrame({
            'open': [100.0] * 4,
            'high': [100.0, 102.0 if side == 1 else 100.0, 100.0, 100.0],
            'low': [100.0, 100.0 if side == 1 else 98.0, 100.0, 100.0],
            'close': [100.0, close, 100.0, 100.0],
            'volume': [1000.0] * 4,
        }, index=index)
        indicators = pd.DataFrame({'atr': 1.0, 'entry': [side, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, side, 0, 0]
        data.loc[:, ['open', 'high', 'low', 'close']] = 100.0
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, side, side, 0]
        assert (result['size'] > 0.0).all()


if __name__ == '__main__':
    _self_check()
