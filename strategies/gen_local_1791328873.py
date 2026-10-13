from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    lookback: int = 24
    entropy_ceiling: float = 0.82
    min_correlation: float = 0.12
    profit_target: float = 0.01
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local_1791328873"

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params):
        return int(params.lookback + 4)

    def indicators(self, data, params):
        if params.lookback < 4 or not 0.0 <= params.entropy_ceiling <= 1.0:
            raise ValueError("Invalid ordinal complexity parameters")
        if not 0.0 < params.min_correlation <= 1.0:
            raise ValueError("min_correlation must be in (0, 1]")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be positive and finite")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be 1 or 2")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lag_1"] = out["return"].shift(1)
        out["lag_2"] = out["return"].shift(2)
        a, b, c = out["lag_2"], out["lag_1"], out["return"]
        out["pattern"] = np.select(
            [(a < b) & (b < c), (a < c) & (c < b),
             (b < a) & (a < c), (b < c) & (c < a),
             (c < a) & (a < b), (c < b) & (b < a)],
            [0, 1, 2, 3, 4, 5], default=np.nan
        )
        entropy = pd.Series(0.0, index=data.index)
        for k in range(6):
            probability = (out["pattern"] == k).astype(float).where(out["pattern"].notna()).rolling(
                params.lookback, min_periods=params.lookback
            ).mean()
            out[f"pattern_probability_{k}"] = probability
            entropy -= probability * np.log(probability.clip(lower=1e-12))
        out["entropy"] = entropy / np.log(6.0)
        out["autocorrelation"] = out["return"].rolling(
            params.lookback, min_periods=params.lookback
        ).corr(out["lag_1"]).replace([np.inf, -np.inf], np.nan)
        out["forecast"] = out["autocorrelation"] * out["return"]
        eligible = (
            out["entropy"].le(params.entropy_ceiling)
            & out["autocorrelation"].abs().ge(params.min_correlation)
            & out["forecast"].notna()
        )
        out["candidate"] = np.sign(out["forecast"].fillna(0.0)).where(eligible, 0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        if params.max_hold_bars not in (1, 2) or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Invalid exit parameters")
        candidate = indicators["candidate"].fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                held = i - entry_bar + 1
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                gain = position * (closes[i] / entry_price - 1.0) if not invalid_entry else np.nan
                if invalid_entry or gain >= params.profit_target or held >= params.max_hold_bars:
                    position = 0
                decisions[i] = position
                continue
            if confirmed[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(confirmed[i])
                entry_bar = i + 1
                entry_price = np.nan
                decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 99.0,
        "close": [100., 100., 102., 100., 100., 100., 100., 100., 100., 100.],
        "volume": 1000.0
    }, index=index)
    strategy, params = GeneratedStrategy(), OrdinalParams()
    candidates = pd.DataFrame({"candidate": [1, 1, 1, 1, 0, -1, -1, -1, -1, 0]}, index=index)
    result = strategy.generate_signals(data, candidates, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    derived = strategy.indicators(data, params)
    assert derived["candidate"].eq(0).all()
    assert strategy.warmup_bars(params) >= params.lookback + 3


if __name__ == "__main__":
    _self_check()
