from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockRepairParams:
    vol_window: int = 20
    pressure_window: int = 12
    shock_z: float = 1.25
    min_pressure: float = 0.5
    repair_fraction: float = 0.35
    atr_window: int = 14
    breakeven_pct: float = 0.0075
    trail_atr: float = 1.4
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[ShockRepairParams]):
    strategy_id = 'gen_local-w2_1791305612'

    @classmethod
    def params_type(cls) -> type[ShockRepairParams]:
        return ShockRepairParams

    @staticmethod
    def warmup_bars(params: ShockRepairParams) -> int:
        return int(max(params.vol_window + params.pressure_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockRepairParams) -> pd.DataFrame:
        if min(params.vol_window, params.pressure_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two.')
        if not (params.shock_z > 0 and params.min_pressure > 0 and 0 < params.repair_fraction < 1):
            raise ValueError('Invalid shock or repair thresholds.')
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['volatility'] = out['return'].rolling(params.vol_window).std(ddof=0).shift(1).replace(0, np.nan)
        valid = out['return'].notna() & out['volatility'].notna()
        large = out['return'].abs() >= params.shock_z * out['volatility']
        out['shock_return'] = out['return'].where(large, 0.0).where(valid)
        out['repair_return'] = out['return'].where(~large, 0.0).where(valid)
        out['shock_drift'] = out['shock_return'].rolling(params.pressure_window).sum()
        out['repair_drift'] = out['repair_return'].rolling(params.pressure_window).sum()
        out['shock_pressure'] = out['shock_drift'].abs() / (out['volatility'] * np.sqrt(params.pressure_window))
        out['repair_fraction'] = out['repair_drift'].abs() / out['shock_drift'].abs().replace(0, np.nan)
        product = out['shock_drift'] * out['repair_drift']
        reinforcement = (product > 0) & (out['repair_fraction'] < 1.0)
        absorption = (product < 0) & out['repair_fraction'].between(params.repair_fraction, 0.85)
        out['regime'] = np.select([reinforcement, absorption], [1, 2], default=0).astype(int)
        direction = pd.Series(np.sign(out['shock_drift']), index=data.index)
        direction = direction.where(out['regime'].eq(1), -direction)
        eligible = out['regime'].ne(0) & out['shock_pressure'].ge(params.min_pressure)
        eligible &= (direction * out['return']).gt(0)
        out['candidate'] = direction.where(eligible, 0).fillna(0).astype(int)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockRepairParams) -> SignalFrame:
        if not (params.breakeven_pct > 0 and params.trail_atr > 0 and 3 <= params.max_hold <= 5):
            raise ValueError('Require positive stop parameters and a three-to-five-session holding limit.')
        candidate = indicators['candidate'].fillna(0).astype(int)
        confirmed = candidate.ne(0) & candidate.eq(candidate.shift(1))
        confirmed &= indicators['regime'].eq(indicators['regime'].shift(1))
        entries = candidate.where(confirmed, 0).to_numpy()
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = pending = age = 0
        entry = best_gain = 0.0
        stop_gain = -np.inf
        armed = False
        for i in range(len(data)):
            # An entry decided yesterday becomes a position at today's open.
            if pending:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    side = pending
                    entry = opens[i]
                    age = 0
                    best_gain = 0.0
                    stop_gain = -np.inf
                    armed = False
                pending = 0
            valid = np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0
            if side:
                age += 1
                if not valid:
                    side = 0
                    continue
                gain = side * (closes[i] - entry)
                best_gain = max(best_gain, gain)
                if best_gain >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed profit coordinates ratchet upward for both directions.
                    # The corresponding short-price stop ratchets downward.
                    stop_gain = max(stop_gain, 0.0, best_gain - params.trail_atr * atrs[i])
                # Stops are evaluated at closes; exits execute next bar.
                if (armed and gain <= stop_gain) or age >= params.max_hold:
                    side = 0
                    continue
                signals[i] = side
            elif valid and entries[i]:
                pending = int(entries[i])
                signals[i] = pending
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = ShockRepairParams(breakeven_pct=0.005, trail_atr=1.0)
    index = pd.date_range('2020-01-01', periods=8)
    for side in (1, -1):
        for trailing in (True, False):
            moves = np.array([0, 0, 1, 2, 0, 0, 0, 0], dtype=float) if trailing else np.zeros(8)
            close = 100.0 + side * moves
            data = pd.DataFrame({'open': 100.0, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': 1000.0}, index=index)
            ind = pd.DataFrame({'candidate': [side, side, 0, 0, 0, 0, 0, 0], 'regime': 1, 'atr': 1.0}, index=index)
            result = GeneratedStrategy.generate_signals(data, ind, None, params).data
            expected = [0, 0, side, side, side, 0, 0, 0] if trailing else [0, 0, side, side, side, side, 0, 0]
            assert result['signal'].tolist() == expected
            assert result['size'].gt(0).all()
            ind.loc[index[1], 'candidate'] = 0
            assert GeneratedStrategy.generate_signals(data, ind, None, params).data['signal'].eq(0).all()
    indicators = GeneratedStrategy.indicators(data, params)
    assert indicators['candidate'].eq(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= params.vol_window + params.pressure_window + 1


if __name__ == '__main__':
    _self_check()
