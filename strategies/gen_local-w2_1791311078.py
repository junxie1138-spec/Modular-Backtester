from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    complexity_window: int = 6
    volatility_window: int = 8
    entropy_floor: float = 0.75
    selling_threshold: float = 0.8
    target_daily_volatility: float = 0.008
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791311078"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        if not 3 <= params.complexity_window <= 7:
            raise ValueError("complexity_window must be between 3 and 7")
        if not 2 <= params.volatility_window <= 9:
            raise ValueError("volatility_window must be between 2 and 9")
        if params.holding_bars not in (1, 2):
            raise ValueError("holding_bars must be 1 or 2")
        if not 0.0 <= params.entropy_floor <= 1.0:
            raise ValueError("entropy_floor must be between 0 and 1")
        if not np.isfinite(params.selling_threshold) or params.selling_threshold <= 0:
            raise ValueError("selling_threshold must be positive and finite")
        if not np.isfinite(params.target_daily_volatility) or params.target_daily_volatility <= 0:
            raise ValueError("target_daily_volatility must be positive and finite")
        return int(max(params.complexity_window + 3, params.volatility_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        r = data["close"].replace(0.0, np.nan).pct_change(fill_method=None)
        r = r.replace([np.inf, -np.inf], np.nan)
        out["return"] = r
        a, b, c = r.shift(2), r.shift(1), r
        valid = a.notna() & b.notna() & c.notna()
        # Equal returns retain chronological order, yielding deterministic ranks.
        code = (4 * (a > b).astype(int) + 2 * (a > c).astype(int) + (b > c).astype(int)).astype(float)
        out["ordinal_code"] = code.where(valid)
        terms = []
        for pattern in (0, 1, 3, 4, 6, 7):
            hits = out["ordinal_code"].eq(pattern).astype(float).where(valid)
            p = hits.rolling(params.complexity_window, min_periods=params.complexity_window).mean()
            out[f"pattern_probability_{pattern}"] = p
            terms.append(p * np.log(p.where(p > 0.0, 1.0)))
        out["entropy"] = -pd.concat(terms, axis=1).sum(axis=1, min_count=6) / np.log(6.0)
        out["signed_drift"] = r.rolling(params.complexity_window, min_periods=params.complexity_window).sum()
        out["return_energy"] = r.pow(2).rolling(params.complexity_window, min_periods=params.complexity_window).sum().pow(0.5)
        out["selling_pressure"] = out["signed_drift"] / out["return_energy"].replace(0.0, np.nan)
        out["volatility"] = r.rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        ready = indicators[["entropy", "selling_pressure", "return", "volatility"]].notna().all(axis=1)
        entry = (
            ready
            & indicators["entropy"].ge(params.entropy_floor)
            & indicators["selling_pressure"].le(-params.selling_threshold)
            & indicators["return"].gt(0.0)
        ).fillna(False).to_numpy(dtype=bool)
        target_size = (params.target_daily_volatility / indicators["volatility"].clip(lower=1e-6)).clip(lower=0.05, upper=0.95)
        target_size = target_size.replace([np.inf, -np.inf], np.nan).fillna(0.05).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.05, dtype=float)
        exit_bar = -1
        entry_size = 0.05
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signal[i] = 1
                size[i] = entry_size
            elif entry[i]:
                exit_bar = i + params.holding_bars
                entry_size = float(target_size[i])
                signal[i] = 1
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"entropy": 1.0, "selling_pressure": -1.0, "return": 0.001, "volatility": np.arange(1, 8) * 0.01}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0]
    assert np.allclose(result["size"], [0.05, 0.8, 0.8, 0.05, 0.2, 0.2, 0.05])
    assert GeneratedStrategy.warmup_bars(None, params) <= 10
    ind.loc[:, "entropy"] = np.nan
    assert GeneratedStrategy.generate_signals(None, data, ind, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
