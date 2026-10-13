from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SlopeEvidenceParams:
    slope_window: int = 20
    innovation_window: int = 40
    evidence_threshold: float = 3.0
    evidence_drift: float = 0.1
    profit_target: float = 0.03
    max_hold_bars: int = 10
    size: float = 0.95

    def __post_init__(self):
        if self.slope_window < 2 or self.innovation_window < 2:
            raise ValueError('Indicator windows must be at least two.')
        if self.evidence_threshold <= 0 or self.evidence_drift < 0:
            raise ValueError('Invalid evidence parameters.')
        if self.profit_target <= 0 or self.max_hold_bars < 1:
            raise ValueError('Invalid exit parameters.')
        if not 0 < self.size <= 1:
            raise ValueError('Size must be in (0, 1].')


class GeneratedStrategy(BaseStrategy[SlopeEvidenceParams]):
    strategy_id = 'gen_local_1790881259'

    @classmethod
    def params_type(cls):
        return SlopeEvidenceParams

    def warmup_bars(self, params: SlopeEvidenceParams) -> int:
        return int(max(200, params.slope_window + params.innovation_window + 1))

    def indicators(self, data: pd.DataFrame, params: SlopeEvidenceParams) -> pd.DataFrame:
        close = data['close'].astype(float)
        log_close = np.log(close.where(close > 0))
        x = np.arange(params.slope_window, dtype=float)
        weights = x - x.mean()
        weights /= np.dot(weights, weights)
        values = np.convolve(log_close.to_numpy(), weights[::-1], mode='full')[:len(data)] if len(data) else np.array([], dtype=float)
        slope = pd.Series(values, index=data.index)
        slope.iloc[:params.slope_window - 1] = np.nan
        revision = slope.diff()
        scale = revision.rolling(params.innovation_window, min_periods=params.innovation_window).std(ddof=0)
        standardized = revision.div(scale.where(scale > 1e-12)).replace([np.inf, -np.inf], np.nan)
        increments = standardized.clip(-3.0, 3.0).fillna(0.0) - params.evidence_drift
        cumulative = increments.cumsum()
        evidence = cumulative - cumulative.cummin().clip(upper=0.0)
        ma200 = close.rolling(200, min_periods=200).mean()
        ready = standardized.notna() & ma200.notna() & slope.notna()
        entry = (
            ready
            & (close > ma200)
            & (slope > 0)
            & (evidence >= params.evidence_threshold)
            & (evidence.shift(1).fillna(0.0) < params.evidence_threshold)
        )
        return pd.DataFrame({
            'log_close': log_close,
            'latent_slope': slope,
            'slope_revision': revision,
            'revision_scale': scale,
            'standardized_revision': standardized,
            'evidence_increment': increments,
            'cumulative_evidence': cumulative,
            'evidence': evidence,
            'ma200': ma200,
            'ready': ready,
            'entry': entry.fillna(False),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SlopeEvidenceParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decision_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if decision_bar >= 0:
                # Record the opening fill only when that bar has arrived.
                if i == decision_bar + 1:
                    entry_price = opens[i]
                age = i - decision_bar
                target_hit = (
                    np.isfinite(entry_price)
                    and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                # Targets are closing decisions; exits execute next bar.
                if target_hit or age >= params.max_hold_bars:
                    decision_bar = -1
                    entry_price = np.nan
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                decision_bar = i
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': float(params.size)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry': [True, False, False, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = SlopeEvidenceParams(max_hold_bars=3)
    timed = strategy.generate_signals(data, indicators, None, params).data
    assert timed['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    data.loc[index[2], 'close'] = 104.0
    targeted = strategy.generate_signals(data, indicators, None, params).data
    assert targeted['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert (targeted['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
