from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    pattern_window: int = 20
    percentile_window: int = 126
    entry_percentile: float = 0.8
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local_1791331711"

    @classmethod
    def params_type(cls) -> type[OrdinalParams]:
        return OrdinalParams

    @staticmethod
    def warmup_bars(params: OrdinalParams) -> int:
        if params.pattern_window < 3 or params.percentile_window < 2:
            raise ValueError("Indicator windows must be at least three and two bars respectively.")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one.")
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive.")
        return int(params.pattern_window + params.percentile_window + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["close_lag2"] = close.shift(2)
        out["close_lag1"] = close.shift(1)
        a, b, c = out["close_lag2"], out["close_lag1"], close
        valid = a.notna() & b.notna() & c.notna()
        # Equal closes are ordered chronologically, giving deterministic ties.
        out["ordinal_pattern"] = (
            4 * (a <= b).astype(int)
            + 2 * (a <= c).astype(int)
            + (b <= c).astype(int)
        ).where(valid)
        probability_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            name = f"pattern_probability_{code}"
            out[name] = (
                out["ordinal_pattern"].eq(code).astype(float).where(valid)
                .rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            )
            probability_columns.append(name)
        probabilities = out[probability_columns]
        out["ordinal_entropy"] = (
            -(probabilities * np.log(probabilities.where(probabilities > 0, 1.0)))
            .sum(axis=1, min_count=6) / np.log(6.0)
        ).clip(0.0, 1.0)
        out["upward_dominance"] = out["pattern_probability_7"] - out["pattern_probability_0"]
        out["order_strength"] = out["upward_dominance"] * (1.0 - out["ordinal_entropy"])
        out["entry_threshold"] = (
            out["order_strength"]
            .rolling(params.percentile_window, min_periods=params.percentile_window)
            .quantile(params.entry_percentile).shift(1)
        )
        out["price_drift"] = close.diff(params.pattern_window)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        eligible = (
            indicators["entry_threshold"].notna()
            & indicators["order_strength"].gt(0.0)
            & indicators["order_strength"].ge(indicators["entry_threshold"])
            & indicators["price_drift"].gt(0.0)
        ).fillna(False).to_numpy(dtype=bool)
        position = np.zeros(len(data), dtype=np.int64)
        exit_bar = -1
        for i, enter in enumerate(eligible):
            if exit_bar >= 0:
                if i < exit_bar:
                    position[i] = 1
                else:
                    # Always emit an exit; reconsider entry on the following bar.
                    exit_bar = -1
            elif enter:
                exit_bar = i + params.holding_bars
                position[i] = 1
        df = pd.DataFrame({"signal": position, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = OrdinalParams(holding_bars=3)
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    entry = np.array([True, False, True, False, False, True, False, False, False, False, False, False])
    derived = pd.DataFrame({"entry_threshold": 0.5, "order_strength": entry.astype(float), "price_drift": 1.0}, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]
    assert result["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
