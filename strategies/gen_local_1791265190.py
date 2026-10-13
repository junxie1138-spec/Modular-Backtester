from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalTrendParams:
    pattern_window: int = 18
    atr_window: int = 14
    complexity_max: float = 0.55
    dominance_min: float = 0.20
    breakeven_pct: float = 0.008
    trail_atr: float = 1.6
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[OrdinalTrendParams]):
    strategy_id = "gen_local_1791265190"

    @classmethod
    def params_type(cls) -> type[OrdinalTrendParams]:
        return OrdinalTrendParams

    @staticmethod
    def warmup_bars(params: OrdinalTrendParams) -> int:
        return int(max(200, params.pattern_window + 3, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: OrdinalTrendParams) -> pd.DataFrame:
        if not (params.pattern_window >= 3 and params.atr_window >= 1
                and 0.0 <= params.complexity_max <= 1.0
                and 0.0 < params.dominance_min <= 1.0
                and params.breakeven_pct > 0.0 and params.trail_atr > 0.0
                and 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid ordinal trend parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna()
        # Stable chronological tie breaking: equal prices retain time order.
        out["pattern"] = (2 * ((a > b).astype(int) + (a > c).astype(int))
                          + (b > c).astype(int)).where(valid)
        out["previous_pattern"] = out["pattern"].shift(1)
        paired = out["pattern"].notna() & out["previous_pattern"].notna()
        joint_entropy = pd.Series(0.0, index=data.index)
        previous_entropy = pd.Series(0.0, index=data.index)
        for previous in range(6):
            prior = out["previous_pattern"].eq(previous).astype(float).where(paired)
            q = prior.rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            out[f"previous_probability_{previous}"] = q
            previous_entropy -= q * np.log(q.clip(lower=1e-12))
            for current in range(6):
                event = (out["previous_pattern"].eq(previous)
                         & out["pattern"].eq(current)).astype(float).where(paired)
                p = event.rolling(params.pattern_window, min_periods=params.pattern_window).mean()
                out[f"transition_probability_{previous}_{current}"] = p
                joint_entropy -= p * np.log(p.clip(lower=1e-12))
        out["joint_entropy"] = joint_entropy
        out["previous_entropy"] = previous_entropy
        out["complexity"] = ((joint_entropy - previous_entropy) / np.log(6.0)).clip(0.0, 1.0)
        for current in range(6):
            event = out["pattern"].eq(current).astype(float).where(paired)
            out[f"pattern_probability_{current}"] = event.rolling(
                params.pattern_window, min_periods=params.pattern_window).mean()
        out["dominance"] = out["pattern_probability_0"] - out["pattern_probability_5"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        ready = np.isfinite(out[["complexity", "dominance", "ma200", "atr"]]).all(axis=1)
        ready &= (out["atr"] > 0.0) & (close > 0.0)
        out["ready"] = ready
        ordered = ready & (out["complexity"] <= params.complexity_max)
        long = ordered & (out["dominance"] >= params.dominance_min) & (close > out["ma200"])
        short = ordered & (out["dominance"] <= -params.dominance_min) & (close < out["ma200"])
        out["entry_direction"] = np.select([long, short], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalTrendParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].to_numpy(dtype=int)
        ready = indicators["ready"].to_numpy(dtype=bool)
        direction, age = 0, 0
        pending = False
        armed = False
        entry = best = stop = entry_atr = 0.0
        for i in range(len(data)):
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                    direction = 0
                    continue
                # The preceding close's decision fills at this bar's open.
                entry = opens[i]
                best = direction * entry
                # Signed price makes both long and short stops ratchet upward.
                stop = best - params.trail_atr * entry_atr
                age, armed = 0, False
            if direction:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0.0 or not ready[i]:
                    direction = 0
                    continue
                signed_close = direction * closes[i]
                best = max(best, signed_close)
                if best - direction * entry >= params.breakeven_pct * entry:
                    armed = True
                if armed:
                    stop = max(stop, direction * entry, best - params.trail_atr * atrs[i])
                # Close-observed stops submit an exit for the next open.
                if signed_close <= stop or age >= params.hold_bars:
                    direction = 0
                    continue
                decisions[i] = direction
            elif ready[i] and entries[i] != 0:
                direction = int(entries[i])
                entry_atr = atrs[i]
                pending = True
                decisions[i] = direction
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    params = OrdinalTrendParams(breakeven_pct=0.005, trail_atr=2.0)
    for direction in (1, -1):
        prices = 100.0 + direction * np.array([0.0, 1.0, 0.4, -0.1, 0.0, 0.0, 0.0, 0.0])
        data = pd.DataFrame({"open": 100.0, "high": prices + 1.0,
                             "low": prices - 1.0, "close": prices, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"atr": 1.0, "ready": True,
                            "entry_direction": [direction, 0, 0, 0, 0, 0, 0, 0]}, index=index)
        result = GeneratedStrategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0, 0, 0, 0]
        data["close"] = 100.0
        result = GeneratedStrategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0, 0, 0]
        assert (result["size"] > 0.0).all()
    index = pd.date_range("2020-01-01", periods=260, freq="B")
    close = 100.0 + np.arange(260) * 0.1
    data = pd.DataFrame({"open": close, "high": close + 0.5,
                         "low": close - 0.5, "close": close, "volume": 1000.0}, index=index)
    ind = GeneratedStrategy.indicators(data, params)
    assert not ind["ready"].iloc[:199].any()
    assert ind["complexity"].iloc[-1] == 0.0
    assert ind["entry_direction"].iloc[-1] == 1
    full = GeneratedStrategy.generate_signals(data, ind, None, params).data
    prefix = data.iloc[:240]
    prefix_ind = GeneratedStrategy.indicators(prefix, params)
    pd.testing.assert_frame_equal(ind.iloc[:240], prefix_ind)
    pd.testing.assert_frame_equal(
        full.iloc[:240],
        GeneratedStrategy.generate_signals(prefix, prefix_ind, None, params).data,
    )


if __name__ == "__main__":
    _self_check()
