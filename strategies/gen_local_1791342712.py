from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TransmissionParams:
    impulse_bars: int = 4
    history_bars: int = 63
    regime_quantile: float = 0.65
    impulse_quantile: float = 0.60


def _reversal_positions(candidate: pd.Series) -> pd.Series:
    positions = np.zeros(len(candidate), dtype=int)
    position = 0
    for i, desired in enumerate(candidate.to_numpy(dtype=int)):
        if position != 0 and desired == -position:
            position = 0
        elif position == 0 and desired != 0:
            position = int(desired)
        positions[i] = position
    return pd.Series(positions, index=candidate.index, dtype=int)


class GeneratedStrategy(BaseStrategy[TransmissionParams]):
    strategy_id = 'gen_local_1791342712'

    @classmethod
    def params_type(cls) -> type[TransmissionParams]:
        return TransmissionParams

    def warmup_bars(self, params: TransmissionParams) -> int:
        return int(params.impulse_bars + params.history_bars + 1)

    def indicators(self, data: pd.DataFrame, params: TransmissionParams) -> pd.DataFrame:
        if params.impulse_bars < 2 or params.history_bars < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not (0.0 < params.regime_quantile < 1.0 and 0.0 < params.impulse_quantile < 1.0):
            raise ValueError('Quantiles must lie strictly between zero and one.')
        n = params.impulse_bars
        h = params.history_bars
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        high = data['high'].where(data['high'] > 0)
        low = data['low'].where(data['low'] > 0)
        out['log_return'] = np.log(close / close.shift(1))
        out['close_vol'] = out['log_return'].rolling(n, min_periods=n).std(ddof=0)
        out['range_variance'] = (np.log(high / low).pow(2) / (4.0 * np.log(2.0))).where(high >= low)
        out['range_vol'] = np.sqrt(out['range_variance'].rolling(n, min_periods=n).mean())
        out['transmission'] = out['close_vol'] / out['range_vol'].replace(0.0, np.nan)
        out['impulse_return'] = out['log_return'].rolling(n, min_periods=n).sum()
        out['impulse'] = out['impulse_return'] / (out['close_vol'].replace(0.0, np.nan) * np.sqrt(n))
        out['regime_threshold'] = out['transmission'].shift(1).rolling(h, min_periods=h).quantile(params.regime_quantile)
        out['impulse_threshold'] = out['impulse'].abs().shift(1).rolling(h, min_periods=h).quantile(params.impulse_quantile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: TransmissionParams) -> SignalFrame:
        required = ['transmission', 'impulse', 'regime_threshold', 'impulse_threshold']
        valid = indicators[required].notna().all(axis=1)
        strong = indicators['impulse'].abs() > indicators['impulse_threshold']
        continuation = indicators['transmission'] > indicators['regime_threshold']
        direction = np.sign(indicators['impulse'].fillna(0.0)).astype(int)
        desired = direction.where(continuation, -direction)
        candidate = desired.where(valid & strong, 0).astype(int)
        df = pd.DataFrame(index=data.index)
        # Neutral observations retain exposure; an opposing entry explicitly exits to flat.
        df['signal'] = _reversal_positions(candidate)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    candidate = pd.Series([1, 0, -1, -1, 0, 1, 1], dtype=int)
    signal = _reversal_positions(candidate).shift(1).fillna(0).astype(int)
    assert signal.tolist() == [0, 1, 1, 0, -1, -1, 0]
