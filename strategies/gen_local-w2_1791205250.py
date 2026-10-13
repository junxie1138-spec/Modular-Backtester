from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    complexity_window: int = 18
    atr_multiple: float = 0.8


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local-w2_1791205250"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return int(max(params.complexity_window + 4, 15))

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        if params.complexity_window < 6:
            raise ValueError("complexity_window must be at least 6")
        if not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("atr_multiple must be finite and positive")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_lag1"] = out["return"].shift(1)
        out["return_lag2"] = out["return"].shift(2)
        a, b, c = out["return_lag2"], out["return_lag1"], out["return"]
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        out["pattern"] = (a.gt(b).astype(int) + 2 * a.gt(c).astype(int) + 4 * b.gt(c).astype(int)).where(valid)

        entropy = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            probability = out["pattern"].eq(code).astype(float).where(valid).rolling(
                params.complexity_window, min_periods=params.complexity_window
            ).mean()
            out[f"pattern_probability_{code}"] = probability
            entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
        out["entropy"] = entropy / np.log(6.0)
        out["entropy_drop"] = out["entropy"].shift(1) - out["entropy"]

        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).replace([np.inf, -np.inf], np.nan)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()

        contraction = out["entropy_drop"].ge(0.03) & out["entropy"].lt(0.90)
        ready = out["atr"].gt(0) & np.isfinite(out["atr"]) & np.isfinite(close)
        upward = c.gt(b) & b.gt(a) & b.gt(0)
        downward = c.lt(b) & b.lt(a) & b.lt(0)
        out["entry"] = np.select(
            [ready & contraction & upward, ready & contraction & downward],
            [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        stop = -np.inf

        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if not valid:
                position = 0
            elif position:
                age += 1
                # The entry fills on this bar after the preceding close decision.
                high_water = close[i] if np.isnan(high_water) else max(high_water, close[i])
                stop = max(stop, high_water - params.atr_multiple * atr[i])
                # The mandated high-water drawdown exit applies to both directions.
                if close[i] <= stop or age >= 2:
                    position = 0
            elif entries[i] != 0:
                position = int(entries[i])
                age = 0
                high_water = np.nan
                stop = -np.inf
            raw[i] = position

        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9)
    close = pd.Series([100, 102, 99, 100, 100, 98, 97, 99, 100], index=index, dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = ShockwaveParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    indicators = pd.DataFrame({"atr": 1.0, "entry": [1, 0, 0, 0, -1, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0]
    assert result["size"].gt(0).all()
    assert strategy.indicators(data, params)["entry"].eq(0).all()
    assert strategy.warmup_bars(params) >= params.complexity_window + 4


if __name__ == "__main__":
    _self_check()
