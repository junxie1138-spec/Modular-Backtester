from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    complexity_window: int = 12
    entropy_floor: float = 0.94
    profit_target: float = 0.007
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791333801"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return int(max(200, params.complexity_window + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.complexity_window < 3 or not 0 <= params.entropy_floor <= 1:
            raise ValueError("Invalid complexity parameters")
        if params.profit_target <= 0 or params.max_hold not in (1, 2):
            raise ValueError("Require a positive target and a one- or two-bar hold")
        out = pd.DataFrame(index=data.index)
        out["intraday_return"] = (
            data["close"] / data["open"].where(data["open"] > 0) - 1
        ).replace([np.inf, -np.inf], np.nan)
        out["lag1"] = out["intraday_return"].shift(1)
        out["lag2"] = out["intraday_return"].shift(2)
        x, y, z = out["lag2"], out["lag1"], out["intraday_return"]
        valid = x.notna() & y.notna() & z.notna()
        # Strict comparisons break ties consistently in favor of older samples.
        out["pattern"] = (
            4 * (y > x).astype(int)
            + 2 * (z > x).astype(int)
            + (z > y).astype(int)
        ).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            probability = (
                out["pattern"].eq(code).astype(float).where(valid)
                .rolling(params.complexity_window, min_periods=params.complexity_window)
                .mean()
            )
            out[f"pattern_probability_{code}"] = probability
            entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
        out["entropy"] = entropy / np.log(6.0)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["friday"] = data.index.dayofweek == 4
        out["entry"] = (
            out["friday"]
            & out["entropy"].ge(params.entropy_floor)
            & out["intraday_return"].lt(0)
            & data["close"].gt(out["ma200"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        if params.profit_target <= 0 or params.max_hold not in (1, 2):
            raise ValueError("Require a positive target and a one- or two-bar hold")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        decision_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if decision_bar >= 0:
                if i == decision_bar + 1:
                    # The prior close's decision fills at this session's open.
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    decision_bar = -1
                    continue
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or i - decision_bar >= params.max_hold:
                    # A close-observed exit fills at the following session's open.
                    decision_bar = -1
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                decision_bar = i
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2024-01-05", periods=4, freq="B")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": [100.0, 100.2, 100.3, 100.0], "volume": 1000.0},
        index=index,
    )
    features = pd.DataFrame({"entry": [True, False, False, False]}, index=index)
    for target, expected in ((0.01, [0, 1, 1, 0]), (0.001, [0, 1, 0, 0])):
        result = GeneratedStrategy.generate_signals(
            bars, features, None, StrategyParams(profit_target=target)
        )
        assert result.data["signal"].tolist() == expected
        assert result.data["size"].gt(0).all()
