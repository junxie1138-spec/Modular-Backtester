from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_threshold: float = 0.10
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local-w2_1790970866'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return 26

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if not (0.0 < params.area_threshold < 0.5):
            raise ValueError('area_threshold must be between zero and 0.5')
        if not (np.isfinite(params.trail_atr) and params.trail_atr > 0.0):
            raise ValueError('trail_atr must be positive and finite')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        high = data['high']
        low = data['low']
        out['bar_range'] = high - low
        out['true_range'] = pd.concat([
            out['bar_range'],
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(20, min_periods=20).mean()
        out['recent_range'] = out['true_range'].rolling(6, min_periods=6).mean()
        out['span'] = high.rolling(6).max() - low.rolling(6).min()
        out['volume_mean'] = data['volume'].rolling(20, min_periods=20).mean()
        out['location'] = ((2.0 * close - high - low) /
                           out['bar_range'].where(out['bar_range'] > 0.0))
        out.loc[out['bar_range'].eq(0.0), 'location'] = 0.0
        out['flow'] = (out['location'] * data['volume'] /
                       out['volume_mean'].where(out['volume_mean'] > 0.0))
        out['return'] = close.pct_change(fill_method=None)
        out['flow_path_length'] = out['flow'].abs().rolling(6).sum()
        out['price_path_length'] = out['return'].abs().rolling(6).sum()
        # Antisymmetric second-level signature of six path increments.
        # Positive area means positive flow preceded positive price movement,
        # or negative price movement preceded positive flow.
        out['raw_area'] = 0.0
        for lag in range(1, 6):
            pairs = (out['flow'].shift(lag) * out['return'] -
                     out['return'].shift(lag) * out['flow'])
            out['raw_area'] += 0.5 * pairs.rolling(6 - lag).sum()
        out['area_scale'] = out['flow_path_length'] * out['price_path_length']
        out['area'] = out['raw_area'] / out['area_scale'].where(out['area_scale'] > 0.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out['compressed'] = ((out['recent_range'] < 0.8 * out['atr']) &
                             (out['span'] < 3.0 * out['atr']) &
                             out['atr'].gt(0.0))
        buy = (out['compressed'] & out['area'].gt(params.area_threshold) &
               out['return'].gt(0.0))
        sell = (out['compressed'] & out['area'].lt(-params.area_threshold) &
                out['return'].lt(0.0))
        out['candidate'] = np.select([buy, sell], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry = np.nan
        stop = -np.inf
        best = -np.inf
        entry_atr = np.nan
        armed = False
        held = 0
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if side:
                if np.isnan(entry):
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        side = 0
                        continue
                    # Next-open entry estimate includes the configured 5 bps slippage.
                    entry = opens[i] * (1.0 + side * 0.0005)
                    best = side * entry
                    stop = best - 2.0 * entry_atr
                held += 1
                if not np.isfinite(close) or close <= 0.0 or not np.isfinite(atr) or atr <= 0.0:
                    side = 0
                else:
                    signed_close = side * close
                    # Stops are evaluated at the close; exit executes next open.
                    if signed_close <= stop or held >= 8:
                        side = 0
                    else:
                        best = max(best, signed_close)
                        if best - side * entry >= 0.01 * entry:
                            armed = True
                        if armed:
                            # Signed-price stop only rises: long stops rise and
                            # short covering stops fall in ordinary price units.
                            stop = max(stop, side * entry, best - params.trail_atr * atr)
                decisions[i] = side
                # An exit consumes this bar; no same-close reentry.
                continue
            if (candidates[i] != 0 and np.isfinite(close) and close > 0.0 and
                    np.isfinite(atr) and atr > 0.0):
                side = int(candidates[i])
                entry = np.nan
                entry_atr = atr
                armed = False
                held = 0
                decisions[i] = side
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7)
    for side in (1, -1):
        # A two-ATR advance arms breakeven; the subsequent retreat exits.
        close = 100.0 + side * np.array([0.0, 0.0, 2.0, 1.0, -0.1, -0.1, -0.1])
        data = pd.DataFrame({'open': 100.0, 'high': close + 0.5,
                             'low': close - 0.5, 'close': close,
                             'volume': 1000.0}, index=index)
        ind = pd.DataFrame({'atr': 1.0,
                            'candidate': [side, 0, 0, 0, 0, 0, 0]}, index=index)
        result = GeneratedStrategy.generate_signals(None, data, ind, None, SignatureParams())
        assert result.data['signal'].tolist() == [0, side, side, side, side, 0, 0]
        assert result.data['size'].gt(0.0).all()


if __name__ == '__main__':
    _self_check()
