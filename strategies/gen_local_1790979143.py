from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    pattern_window: int = 36
    entropy_floor: float = 0.85
    min_reversion: float = 0.10
    profit_target: float = 0.006
    max_hold: int = 2
    min_size: float = 0.25


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790979143"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        if params.pattern_window < 6 or params.max_hold not in (1, 2):
            raise ValueError("pattern_window must be >= 6; max_hold must be 1 or 2")
        if not (0.0 <= params.entropy_floor <= 1.0 and 0.0 < params.min_reversion < 1.0):
            raise ValueError("Invalid entropy or autocorrelation threshold")
        if not (0.0 < params.profit_target < 1.0 and 0.0 < params.min_size <= 1.0):
            raise ValueError("Invalid target or position size")
        return int(params.pattern_window + 3)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        self.warmup_bars(params)
        w = params.pattern_window
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].where(data["close"] > 0).pct_change(
            fill_method=None
        ).replace([np.inf, -np.inf], np.nan)
        out["lag1"] = out["return"].shift(1)
        out["lag2"] = out["return"].shift(2)
        a, b, c = out["lag2"], out["lag1"], out["return"]
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        # Three pairwise order comparisons encode the six strict permutations.
        out["pattern"] = (
            4 * (a < b).astype(int) + 2 * (a < c).astype(int) + (b < c).astype(int)
        ).astype(float).where(valid)
        probability_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            name = "prob_" + str(code)
            out[name] = out["pattern"].eq(code).astype(float).where(valid).rolling(
                w, min_periods=w
            ).mean()
            probability_columns.append(name)
        probabilities = out[probability_columns]
        out["entropy"] = -(
            probabilities * np.log(probabilities.where(probabilities > 0, 1.0))
        ).sum(axis=1, min_count=6) / np.log(6.0)
        out["autocorrelation"] = c.rolling(w, min_periods=w).corr(b).replace(
            [np.inf, -np.inf], np.nan
        ).clip(-1.0, 1.0)
        out["volatility"] = c.rolling(w, min_periods=w).std().replace(0.0, np.nan)
        out["downside_z"] = (-c / out["volatility"]).replace(
            [np.inf, -np.inf], np.nan
        )
        out["strength"] = (
            out["entropy"].clip(0.0, 1.0)
            * (-out["autocorrelation"]).clip(0.0, 1.0)
            * out["downside_z"].clip(0.0, 2.0) / 2.0
        ).fillna(0.0).clip(0.0, 1.0)
        out["eligible"] = (
            (out["entropy"] >= params.entropy_floor)
            & (out["autocorrelation"] <= -params.min_reversion)
            & (out["downside_z"] > 0.0)
            & (out["downside_z"] <= 2.5)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        pending_entry = False
        entry_bar = -1
        entry_price = 0.0
        exposure = params.min_size
        for i in range(n):
            # The previous close's decision enters at this bar's open.
            if pending_entry:
                pending_entry = False
                if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                    continue
                entry_bar = i
                entry_price = opens[i]
            if entry_bar >= 0:
                sizes[i] = exposure
                target_hit = np.isfinite(closes[i]) and closes[i] >= (
                    entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.max_hold
                if target_hit or expired or not np.isfinite(closes[i]):
                    # Close-confirmed exit is executed at the next open.
                    entry_bar = -1
                else:
                    signals[i] = 1
            elif eligible[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                exposure = params.min_size + (1.0 - params.min_size) * strength[i]
                sizes[i] = exposure
                signals[i] = 1
                pending_entry = True
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": [100.0, 100.8, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    features = pd.DataFrame({
        "eligible": [True, False, False, True, False, False, False],
        "strength": [0.4, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
    }, index=index)
    strategy, params = GeneratedStrategy(), StrategyParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 1, 1, 0]
    assert np.allclose(result["size"].iloc[[1, 4, 5]], [0.55, 0.85, 0.85])
    assert not strategy.generate_signals(
        data, strategy.indicators(data, params), ctx, params
    ).data["signal"].any()
    for end in range(1, len(data) + 1):
        prefix = strategy.generate_signals(
            data.iloc[:end], features.iloc[:end], ctx, params
        ).data
        assert prefix.equals(result.iloc[:end])


if __name__ == "__main__":
    _self_check()
