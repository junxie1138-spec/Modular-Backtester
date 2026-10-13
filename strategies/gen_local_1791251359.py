from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    pattern_window: int = 40
    contraction_lag: int = 5
    entropy_ceiling: float = 0.80
    entropy_drop: float = 0.06
    correlation_floor: float = 0.10
    atr_window: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local_1791251359"

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params):
        return int(max(params.pattern_window + params.contraction_lag + 4,
                       params.atr_window + 1))

    def indicators(self, data, params):
        if (params.pattern_window < 6 or params.contraction_lag < 1
                or params.atr_window < 1 or params.holding_bars < 1
                or not 0 < params.entropy_ceiling <= 1
                or not 0 <= params.correlation_floor < 1
                or params.entropy_drop <= 0 or params.stop_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        r = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        triples = np.column_stack([r.shift(2), r.shift(1), r])
        order = np.argsort(triples, axis=1, kind="stable")
        pattern = pd.Series(order[:, 0] * 2 + (order[:, 1] > order[:, 2]),
                            index=data.index, dtype=float)
        pattern = pattern.where(np.isfinite(triples).all(axis=1))
        transition = 6 * pattern.shift(1) + pattern
        probabilities = pd.DataFrame({
            code: transition.eq(code).astype(float).where(transition.notna())
                  .rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            for code in range(36)
        }, index=data.index)
        joint_entropy = -(probabilities * np.log(probabilities.where(probabilities > 0))).sum(axis=1, min_count=1)
        origins = pd.DataFrame({
            state: probabilities.iloc[:, state * 6:(state + 1) * 6].sum(axis=1, min_count=6)
            for state in range(6)
        }, index=data.index)
        origin_entropy = -(origins * np.log(origins.where(origins > 0))).sum(axis=1, min_count=1)
        # Overlapping three-return patterns have at most three successors.
        complexity = ((joint_entropy - origin_entropy) / np.log(3.0)).clip(0, 1)
        out["return"] = r
        out["pattern"] = pattern
        out["transition"] = transition
        out["complexity"] = complexity
        out["contraction"] = complexity.shift(params.contraction_lag) - complexity
        out["autocorrelation"] = r.rolling(params.pattern_window).corr(r.shift(1)).clip(-1, 1)
        out["drift"] = r.rolling(params.pattern_window).mean()
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["conviction"] = (
            ((params.entropy_ceiling - complexity) / params.entropy_ceiling).clip(0, 1)
            * ((out["autocorrelation"] - params.correlation_floor)
               / (1 - params.correlation_floor)).clip(0, 1)
        ).fillna(0)
        out["entry_ready"] = (
            (complexity < params.entropy_ceiling)
            & (out["contraction"] >= params.entropy_drop)
            & (out["autocorrelation"] > params.correlation_floor)
            & (out["drift"] > 0) & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].fillna(0).clip(0, 1).to_numpy()
        atr = indicators["atr"].to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        active = False
        pending = False
        stop = 0.0
        entry_bar = 0
        entry_size = 0.20
        for i in range(len(data)):
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    active = False
                    continue
                stop = opens[i] - params.stop_atr * atr[i - 1]
                entry_bar = i
            if active:
                if (not np.isfinite(closes[i]) or closes[i] < stop
                        or i - entry_bar + 1 >= params.holding_bars):
                    active = False
                else:
                    signal[i] = 1
                    sizes[i] = entry_size
            elif ready[i] and (i == 0 or not ready[i - 1]):
                if np.isfinite(atr[i]) and atr[i] > 0:
                    active = pending = True
                    entry_size = 0.20 + 0.75 * conviction[i]
                    signal[i] = 1
                    sizes[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": 100.0, "high": 111.0, "low": 94.0,
        "close": [100.0, 110.0, 99.0, 95.0, 100.0, 100.0], "volume": 1000.0
    }, index=index)
    indicators = pd.DataFrame({
        "entry_ready": [True, False, False, False, False, False],
        "conviction": 0.8, "atr": 2.0
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, OrdinalParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert np.allclose(result.loc[result["signal"] == 1, "size"], 0.8)
    assert (result["size"] > 0).all()
    timed = strategy.generate_signals(data, indicators, None, OrdinalParams(holding_bars=2)).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
