from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    activation_pct: float = 0.008
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791276820"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        # Prior 20-session range baseline, followed by a two-session shift.
        return 23

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if not np.isfinite(params.activation_pct) or params.activation_pct <= 0:
            raise ValueError("activation_pct must be positive and finite")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        high, low, close = data['high'], data['low'], data['close']
        out['range'] = (high - low).where(high > low)
        out['range_baseline'] = out['range'].rolling(20, min_periods=20).median().shift(1)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['peak'] = close.rolling(20, min_periods=20).max()
        out['drawdown'] = 1.0 - close / out['peak'].where(out['peak'] > 0)
        out['close_location'] = (close - low) / out['range']
        out['shock'] = (
            (out['range'] >= 1.8 * out['range_baseline'])
            & (close < previous_close)
            & (out['close_location'] <= 0.30)
            & (out['drawdown'] >= 0.02)
        )
        out['absorption'] = (
            (high <= high.shift(1))
            & (low >= low.shift(1))
            & (out['range'] <= 0.65 * out['range'].shift(1))
        )
        out['reopening'] = (
            (low > low.shift(1))
            & (out['range'] > out['range'].shift(1))
            & (close > close.shift(1))
            & (close > data['open'])
            & (out['close_location'] >= 0.75)
        )
        ready = pd.Series(np.arange(len(data)) >= 22, index=data.index)
        valid = pd.Series(
            np.isfinite(data[['open', 'high', 'low', 'close']].to_numpy()).all(axis=1),
            index=data.index,
        )
        out['entry'] = (
            out['shock'].shift(2, fill_value=False)
            & out['absorption'].shift(1, fill_value=False)
            & out['reopening']
            & out['atr'].gt(0)
            & valid
            & ready
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry_price = stop = peak_close = 0.0
        entry_bar = 0

        for i in range(len(data)):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    armed = False
                    entry_price = opens[i]
                    entry_bar = i
                    peak_close = entry_price
                    stop = entry_price - params.trail_atr * atr[i - 1]

            if active:
                # Only the stop established before this session is tested.
                # A breach requests liquidation at the following session's open.
                breached = np.isfinite(lows[i]) and lows[i] <= stop
                expired = i - entry_bar + 1 >= 4
                if breached or expired:
                    active = False
                    continue
                if np.isfinite(closes[i]):
                    peak_close = max(peak_close, closes[i])
                    # Close-confirmed activation avoids assuming intrabar order.
                    if closes[i] >= entry_price * (1.0 + params.activation_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry_price)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, peak_close - params.trail_atr * atr[i])
                raw[i] = 1
            elif entries[i]:
                raw[i] = 1
                pending = True

        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=30, freq='B')
    data = pd.DataFrame({
        'open': 100.0, 'high': 100.5, 'low': 99.5,
        'close': 100.0, 'volume': 1000000.0,
    }, index=index)
    strategy = object.__new__(GeneratedStrategy)
    params = RecoveryParams()
    calculated = strategy.indicators(data, params)
    assert calculated.index.equals(data.index)
    assert not calculated['entry'].any()
    indicators = pd.DataFrame({'atr': 1.0, 'entry': False}, index=index)
    indicators.loc[index[23], 'entry'] = True
    data.loc[index[24], ['high', 'low', 'close']] = [101.2, 99.5, 101.0]
    data.loc[index[25], ['high', 'low', 'close']] = [102.3, 100.2, 102.0]
    data.loc[index[26], ['high', 'low', 'close']] = [101.5, 100.4, 101.0]
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[23:28].tolist() == [0, 1, 1, 1, 0]
    assert set(result['signal'].unique()) <= {0, 1}
    assert result['size'].gt(0).all()
    # With activation deferred, the four-session expiry remains binding.
    result = strategy.generate_signals(
        data, indicators, None, RecoveryParams(activation_pct=0.10)
    ).data
    assert result['signal'].iloc[23:29].tolist() == [0, 1, 1, 1, 1, 0]


if __name__ == '__main__':
    _self_check()
