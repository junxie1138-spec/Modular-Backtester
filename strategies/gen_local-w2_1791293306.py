from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 20
    vol_window: int = 20
    coarse_threshold: float = 0.55
    migration_threshold: float = 0.02
    target_vol: float = 0.12
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = 'gen_local-w2_1791293306'

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(params.energy_window + 20, params.vol_window + 1, 15))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        log_price = np.log(data['close'].where(data['close'] > 0))
        for scale in (2, 4, 8):
            mean = log_price.rolling(scale, min_periods=scale).mean()
            detail = np.sqrt(scale / 2.0) * (mean - mean.shift(scale))
            out[f'detail_{scale}'] = detail
            out[f'energy_{scale}'] = detail.pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        total_energy = out[['energy_2', 'energy_4', 'energy_8']].sum(axis=1, min_count=3)
        out['coarse_share'] = out['energy_8'] / total_energy.where(total_energy > 0)
        out['migration'] = out['coarse_share'] - out['coarse_share'].shift(5)
        out['coherence'] = (
            out['detail_8'] / np.sqrt(out['energy_8'].where(out['energy_8'] > 0))
        ).clip(0.0, 1.0)
        returns = log_price.diff()
        out['annual_vol'] = returns.rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=0) * np.sqrt(252.0)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['strength'] = (out['coarse_share'] * out['coherence']).clip(0.0, 1.0)
        out['exposure'] = (
            params.target_vol / out['annual_vol'].where(out['annual_vol'] > 0)
            * (0.15 + 0.85 * out['strength'])
        ).clip(0.02, 0.95)
        required = out[['migration', 'coarse_share', 'coherence', 'annual_vol', 'atr', 'exposure']]
        ready = np.isfinite(required).all(axis=1) & (out['annual_vol'] > 0) & (out['atr'] > 0)
        out['ready'] = ready.astype(int)
        out['eligible'] = (
            ready
            & (out['coarse_share'] >= params.coarse_threshold)
            & (out['migration'] >= params.migration_threshold)
            & (out['detail_8'] > 0)
            & (out['coherence'] >= 0.25)
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.02, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        eligible = indicators['eligible'].to_numpy(dtype=bool)
        ready = indicators['ready'].to_numpy(dtype=bool)
        exposure = indicators['exposure'].to_numpy(dtype=float)
        active = False
        pending = False
        armed = False
        entry = 0.0
        stop = 0.0
        entry_bar = 0
        for i in range(count):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    armed = False
                    entry = opens[i]
                    entry_bar = i
                    stop = entry - params.trail_atr * atr[i - 1]
            if active:
                if not ready[i] or not np.isfinite(closes[i]) or closes[i] <= 0:
                    active = False
                    continue
                # Stops are assessed at closes; any exit fills on the next bar.
                if closes[i] <= stop or i - entry_bar + 1 >= params.max_hold:
                    active = False
                    continue
                if closes[i] >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, closes[i] - params.trail_atr * atr[i])
                signals[i] = 1
                sizes[i] = float(np.clip(exposure[i], 0.02, 0.95))
            elif eligible[i]:
                pending = True
                signals[i] = 1
                sizes[i] = float(np.clip(exposure[i], 0.02, 0.95))
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.02).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({
        'open': [100.0] * 5,
        'high': [101.0, 105.0, 104.0, 102.0, 101.0],
        'low': [99.0, 99.0, 102.0, 100.0, 99.0],
        'close': [100.0, 104.0, 103.0, 100.5, 100.0],
        'volume': [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0, 1.0, 4.0, 1.0, 1.0],
        'eligible': [1, 0, 0, 0, 0],
        'ready': [1] * 5,
        'exposure': [0.4, 0.5, 0.6, 0.7, 0.8],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, WaveletParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0]
    assert np.allclose(result['size'].to_numpy(), [0.02, 0.4, 0.5, 0.6, 0.02])
    indicators['atr'] = 1.0
    data['close'] = 100.0
    result = strategy.generate_signals(data, indicators, None, WaveletParams(max_hold=2)).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    derived = strategy.indicators(data, WaveletParams())
    assert np.isfinite(derived.to_numpy()).all()
    assert derived['eligible'].sum() == 0


if __name__ == '__main__':
    _self_check()
