from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    volume_multiple: float = 1.10
    stop_atr: float = 1.00

    def __post_init__(self):
        if not np.isfinite(self.volume_multiple) or self.volume_multiple <= 0:
            raise ValueError('volume_multiple must be finite and positive')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local_1791305820'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        # Prior-volume baseline: 21 bars; outcome maturity: 3;
        # historical window: 252. This covers the entire dependency chain.
        return 276

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous = close.shift(1)
        out['change'] = close - previous
        out['move'] = np.sign(out['change'])
        out['phase'] = np.minimum((data.index.day - 1) // 7, 3)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['volume_baseline'] = data['volume'].rolling(20, min_periods=20).median().shift(1)
        out['event'] = (
            (out['volume_baseline'] > 0)
            & (data['volume'] >= params.volume_multiple * out['volume_baseline'])
            & (out['atr'] > 0)
            & (out['change'].abs() >= 0.25 * out['atr'])
            & np.isfinite(out['move'])
        )

        # At bar i, the event at i-3 has completed its two-session
        # open-to-open holding interval: entry i-2, exit i.
        out['matured_phase'] = out['phase'].shift(3)
        out['matured_event'] = out['event'].shift(3).eq(True)
        out['matured_return'] = (
            data['open'] / data['open'].shift(2).replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        out['matured_payoff'] = out['move'].shift(3) * out['matured_return']
        out['edge'] = np.nan
        for phase in range(4):
            observed = (
                out['matured_event']
                & out['matured_phase'].eq(phase)
                & np.isfinite(out['matured_payoff'])
            )
            count = observed.astype(float).rolling(252, min_periods=252).sum()
            total = out['matured_payoff'].where(observed, 0.0).rolling(252, min_periods=252).sum()
            out[f'count_{phase}'] = count
            out[f'edge_{phase}'] = (total / count.replace(0, np.nan)).where(count >= 4)
            selected = out['phase'].eq(phase)
            out.loc[selected, 'edge'] = out.loc[selected, f'edge_{phase}']
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        eligible = (
            indicators['event'].eq(True)
            & np.isfinite(indicators['edge'])
            & indicators['edge'].gt(0)
            & np.isfinite(indicators['atr'])
            & indicators['atr'].gt(0)
        )
        candidates = indicators['move'].where(eligible, 0).fillna(0).to_numpy()
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        held = 0
        entry = np.nan
        distance = np.nan

        for i in range(len(data)):
            if side:
                # The preceding close requested entry; this bar supplies
                # the actual entry open. ATR is frozen at the decision bar.
                if held == 0:
                    entry = opens[i]
                held += 1
                valid = np.isfinite(entry) and entry > 0 and np.isfinite(closes[i])
                stopped = valid and (
                    (side == 1 and closes[i] < entry - distance)
                    or (side == -1 and closes[i] > entry + distance)
                )
                if not valid or stopped or held >= 2:
                    side = 0
                    held = 0
                raw[i] = side
                # Exit decisions cannot also request a replacement entry.
                continue
            if candidates[i] != 0:
                side = int(candidates[i])
                held = 0
                distance = params.stop_atr * atr[i]
                raw[i] = side

        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6, freq='B')
    data = pd.DataFrame({
        'open': 100.0, 'high': 103.0, 'low': 97.0,
        'close': [100.0, 98.0, 100.0, 100.0, 102.0, 100.0],
        'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'event': [True, False, False, True, False, False],
        'move': [1, 0, 0, -1, 0, 0],
        'edge': 0.01, 'atr': 1.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=276)
    stopped = strategy.generate_signals(data, indicators, ctx, params).data
    assert stopped['signal'].tolist() == [0, 1, 0, 0, -1, 0]
    data['close'] = 100.0
    timed = strategy.generate_signals(data, indicators, ctx, params).data
    assert timed['signal'].tolist() == [0, 1, 1, 0, -1, -1]
    assert timed['size'].gt(0).all()

    index = pd.date_range('2020-01-01', periods=400, freq='B')
    prices = 100.0 + np.arange(400) * 0.1
    history = pd.DataFrame({
        'open': prices, 'high': prices + 1.0, 'low': prices - 1.0,
        'close': prices + 0.6, 'volume': np.where(np.arange(400) % 3 == 0, 2000.0, 1000.0),
    }, index=index)
    full = strategy.indicators(history, params)
    prefix = strategy.indicators(history.iloc[:330], params)
    pd.testing.assert_frame_equal(full.iloc[:330], prefix)
    assert strategy.warmup_bars(params) >= 276


if __name__ == '__main__':
    _self_check()
