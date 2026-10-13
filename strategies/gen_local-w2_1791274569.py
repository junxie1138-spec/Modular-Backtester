from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    pattern_window: int = 12
    drawdown_window: int = 40
    min_drawdown: float = 0.015
    entropy_drop: float = 0.035
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791274569"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    @staticmethod
    def warmup_bars(params: RecoveryParams) -> int:
        return int(max(params.drawdown_window + 1, params.pattern_window + 4))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if params.pattern_window < 3 or params.drawdown_window < 3:
            raise ValueError("Lookbacks must be at least three bars.")
        if not 0.0 < params.min_drawdown < 1.0:
            raise ValueError("min_drawdown must be between zero and one.")
        if not 0.0 < params.entropy_drop < 1.0:
            raise ValueError("entropy_drop must be between zero and one.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna()
        # Equal prices retain chronological order, giving deterministic ranks.
        out["ordinal_pattern"] = (
            4 * (a > b).astype(int)
            + 2 * (a > c).astype(int)
            + (b > c).astype(int)
        ).where(valid)

        entropy_terms = []
        for code in (0, 1, 3, 4, 6, 7):
            probability = (out["ordinal_pattern"] == code).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
            out[f"pattern_probability_{code}"] = probability
            entropy_terms.append(-probability * np.log(probability.clip(lower=1e-12)))
        out["entropy"] = pd.concat(entropy_terms, axis=1).sum(
            axis=1, min_count=6
        ) / np.log(6.0)
        out["entropy_decline"] = -out["entropy"].diff()
        out["closing_peak"] = close.rolling(
            params.drawdown_window, min_periods=params.drawdown_window
        ).max()
        out["drawdown"] = close / out["closing_peak"] - 1.0
        out["recovering"] = (close > close.shift(1)).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        qualified = (
            indicators["drawdown"].le(-params.min_drawdown)
            & indicators["entropy_decline"].ge(params.entropy_drop)
            & indicators["recovering"].fillna(False)
        ).fillna(False)
        confirmed = qualified & qualified.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(confirmed.to_numpy(dtype=bool)):
            if i < exit_at:
                signal[i] = 1
            elif i > exit_at and enter:
                exit_at = i + params.hold_bars
                signal[i] = 1
            # At exit_at the target is zero, regardless of confirmation.
        df["signal"] = signal
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame(
        {"drawdown": -0.03, "entropy_decline": 0.05, "recovering": True},
        index=index,
    )
    strategy = GeneratedStrategy()
    for hold, expected in (
        (2, [0, 0, 1, 1, 0, 1, 1, 0]),
        (1, [0, 0, 1, 0, 1, 0, 1, 0]),
    ):
        result = strategy.generate_signals(
            data, indicators, None, RecoveryParams(hold_bars=hold)
        ).data
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0.0).all()
    indicators.loc[index[2]:, "recovering"] = False
    result = strategy.generate_signals(data, indicators, None, RecoveryParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
