from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    entropy_window: int = 20
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791195458"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if params.entropy_window < 6 or params.holding_bars < 1:
            raise ValueError("entropy_window must be >= 6 and holding_bars >= 1")
        return int(max(params.entropy_window + 7, 26))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["range"] = ((data["high"] - data["low"]) / close).replace([np.inf, -np.inf], np.nan)
        out["range"] = out["range"].where(out["range"] >= 0)
        out["range_lag2"] = out["range"].shift(2)
        out["range_lag1"] = out["range"].shift(1)
        a, b, c = out["range_lag2"], out["range_lag1"], out["range"]
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        out["pattern"] = (4 * a.gt(b).astype(int) + 2 * a.gt(c).astype(int) + b.gt(c).astype(int)).where(valid)
        terms = []
        for code in (0, 1, 3, 4, 6, 7):
            name = f"probability_{code}"
            out[name] = out["pattern"].eq(code).astype(float).where(valid).rolling(params.entropy_window, min_periods=params.entropy_window).mean()
            probability = out[name]
            terms.append(probability * np.log(probability.where(probability > 0, 1.0)))
        out["entropy"] = -pd.concat(terms, axis=1).sum(axis=1, min_count=6) / np.log(6.0)
        out["prior_entropy"] = out["entropy"].shift(5)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["daily_volatility"] = out["return"].rolling(20, min_periods=20).std(ddof=0)
        out["annual_volatility"] = out["daily_volatility"] * np.sqrt(252.0)
        out["positive_drift"] = out["return"].rolling(5, min_periods=5).sum()
        out["shock_threshold"] = 3.0 * out["daily_volatility"].shift(1)
        shock_valid = out["return"].notna() & out["shock_threshold"].notna()
        out["shock"] = out["return"].abs().gt(out["shock_threshold"]).astype(float).where(shock_valid)
        out["recent_shock"] = out["shock"].rolling(5, min_periods=5).max()
        out["target_size"] = (0.10 / out["annual_volatility"].where(out["annual_volatility"] > 0)).clip(0.05, 0.95)
        out["entry"] = (
            out["entropy"].lt(0.80)
            & out["prior_entropy"].gt(0.90)
            & out["positive_drift"].gt(0)
            & out["recent_shock"].eq(0)
            & out["target_size"].notna()
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        sizes = indicators["target_size"].replace([np.inf, -np.inf], np.nan).fillna(0.05).clip(0.05, 0.95).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.05, dtype=float)
        remaining = 0
        entry_size = 0.05
        for i in range(len(data)):
            if remaining > 0:
                remaining -= 1
            elif entries[i]:
                remaining = params.holding_bars
                entry_size = sizes[i]
            if remaining > 0:
                signal[i] = 1
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=32, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = Params()
    indicators = strategy.indicators(data, params)
    assert not indicators["entry"].any()
    indicators["entry"] = False
    indicators.loc[index[[3, 5, 11, 12]], "entry"] = True
    indicators["target_size"] = 0.4
    result = strategy.generate_signals(data, indicators, None, params).data
    expected = list(range(4, 12)) + list(range(13, 21))
    assert np.flatnonzero(result["signal"].to_numpy()).tolist() == expected
    assert result.loc[result["signal"].eq(1), "size"].eq(0.4).all()
    assert result["size"].gt(0).all()
