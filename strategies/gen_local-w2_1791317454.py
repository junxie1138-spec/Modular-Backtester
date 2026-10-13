from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    complexity_window: int = 20
    percentile_window: int = 126
    tail_percentile: float = 0.25
    momentum_bars: int = 5
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local-w2_1791317454"

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return max(params.complexity_window + params.percentile_window + 3,
                   params.momentum_bars + 1)

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        if (params.complexity_window < 3 or params.percentile_window < 3
                or params.momentum_bars < 1 or params.max_hold_bars < 1
                or not 0 < params.tail_percentile < 0.5
                or not np.isfinite(params.profit_target) or params.profit_target <= 0):
            raise ValueError("Invalid ordinal strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out["lag2"] = out["return"].shift(2)
        out["lag1"] = out["return"].shift(1)
        a, b, c = out["lag2"], out["lag1"], out["return"]
        valid = a.notna() & b.notna() & c.notna()
        # Chronological tie-breaking keeps each triplet in one of six orders.
        out["pattern"] = (4 * (a <= b).astype(int)
                          + 2 * (a <= c).astype(int)
                          + (b <= c).astype(int)).where(valid)
        terms = []
        for pattern in (0, 1, 3, 4, 6, 7):
            name = f"frequency_{pattern}"
            out[name] = out["pattern"].eq(pattern).astype(float).where(valid).rolling(
                params.complexity_window, min_periods=params.complexity_window).mean()
            terms.append(-out[name] * np.log(out[name].clip(lower=1e-12)))
        out["complexity"] = pd.concat(terms, axis=1).sum(
            axis=1, min_count=6) / np.log(6.0)
        history = out["complexity"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window)
        out["lower"] = history.quantile(params.tail_percentile)
        out["upper"] = history.quantile(1.0 - params.tail_percentile)
        out["drift"] = data["close"].pct_change(
            params.momentum_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        ready = out[["complexity", "lower", "upper", "drift"]].notna().all(axis=1)
        ordered = ready & (out["complexity"] < out["lower"])
        disordered = ready & (out["complexity"] > out["upper"])
        out["regime"] = pd.Series(np.select(
            [ordered, disordered], [1.0, -1.0], default=np.nan),
            index=data.index).ffill().fillna(0).astype(int)
        out["direction"] = (np.sign(out["drift"]).fillna(0)
                            * out["regime"]).where(ready, 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        directions = indicators["direction"].to_numpy(dtype=int)
        position, held = 0, 0
        entry = np.nan
        for i in range(len(data)):
            if position:
                # A decision made yesterday fills at today's open.
                if np.isnan(entry):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    entry = opens[i]
                held += 1
                profit = (position * (closes[i] / entry - 1.0)
                          if np.isfinite(closes[i]) else -np.inf)
                if profit >= params.profit_target or held >= params.max_hold_bars:
                    position, held, entry = 0, 0, np.nan
                decisions[i] = position
                continue
            if np.isfinite(closes[i]) and closes[i] > 0 and directions[i]:
                position = int(directions[i])
                held, entry = 0, np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = OrdinalParams(max_hold_bars=3)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    for direction in (1, -1):
        features = pd.DataFrame({"direction": [direction, 0, 0, 0, 0, 0]}, index=index)
        signals = strategy.generate_signals(data, features, None, params).data
        assert signals["signal"].tolist() == [0, direction, direction, direction, 0, 0]
        targeted = data.copy()
        targeted.loc[index[1], "close"] = 103.0 if direction == 1 else 97.0
        signals = strategy.generate_signals(targeted, features, None, params).data
        assert signals["signal"].tolist() == [0, direction, 0, 0, 0, 0]
    index = pd.date_range("2020-01-01", periods=220)
    close = 100.0 * np.exp(np.cumsum(0.01 * np.sin(np.arange(220) * 0.7)))
    data = pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1000.0}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:190], params)
    pd.testing.assert_frame_equal(full.iloc[:190], prefix)
    assert set(full["pattern"].dropna().unique()) <= {0, 1, 3, 4, 6, 7}
    assert full["complexity"].dropna().between(-1e-10, 1.0 + 1e-10).all()
    assert full["direction"].iloc[:strategy.warmup_bars(params) - 1].eq(0).all()


if __name__ == "__main__":
    _self_check()
