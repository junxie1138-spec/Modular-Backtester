from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationDebtParams:
    slope_window: int = 20
    pressure_window: int = 18
    volatility_window: int = 60
    cooling_window: int = 5
    entry_threshold: float = 2.0
    cooling_ratio: float = 0.85


class GeneratedStrategy(BaseStrategy[InnovationDebtParams]):
    strategy_id = "gen_local-w2_1791271490"

    @classmethod
    def params_type(cls) -> type[InnovationDebtParams]:
        return InnovationDebtParams

    @staticmethod
    def warmup_bars(params: InnovationDebtParams) -> int:
        return int(max(
            200,
            max(params.slope_window, params.volatility_window)
            + 2 + max(params.pressure_window, 2 * params.cooling_window) - 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: InnovationDebtParams) -> pd.DataFrame:
        windows = (
            params.slope_window, params.pressure_window,
            params.volatility_window, params.cooling_window,
        )
        if any(w < 2 for w in windows):
            raise ValueError("Indicator windows must be at least two bars")
        if not np.isfinite(params.entry_threshold) or params.entry_threshold <= 0:
            raise ValueError("entry_threshold must be positive and finite")
        if not np.isfinite(params.cooling_ratio) or not 0 < params.cooling_ratio < 1:
            raise ValueError("cooling_ratio must lie between zero and one")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["log_return"] = out["log_close"].diff()
        out["latent_slope"] = out["log_return"].rolling(
            params.slope_window, min_periods=params.slope_window
        ).mean().shift(1)
        out["return_scale"] = out["log_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0).shift(1).replace(0.0, np.nan)
        out["innovation"] = out["log_return"] - out["latent_slope"]
        out["standardized_innovation"] = (
            out["innovation"] / out["return_scale"]
        ).replace([np.inf, -np.inf], np.nan)
        out["innovation_debt"] = out["standardized_innovation"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum() / np.sqrt(params.pressure_window)
        out["shock_intensity"] = out["standardized_innovation"].abs()
        out["recent_intensity"] = out["shock_intensity"].rolling(
            params.cooling_window, min_periods=params.cooling_window
        ).mean()
        out["previous_intensity"] = out["recent_intensity"].shift(params.cooling_window)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["cooling"] = (
            out["recent_intensity"] < params.cooling_ratio * out["previous_intensity"]
        ).fillna(False)
        out["long_setup"] = (
            (out["innovation_debt"] <= -params.entry_threshold) & out["cooling"]
        ).fillna(False)
        out["short_setup"] = (
            (out["innovation_debt"] >= params.entry_threshold) & out["cooling"]
        ).fillna(False)
        out["above_ma"] = (close > out["ma200"]).fillna(False)
        out["below_ma"] = (close < out["ma200"]).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: InnovationDebtParams,
    ) -> SignalFrame:
        columns = ["long_setup", "short_setup", "above_ma", "below_ma"]
        conditions = indicators[columns].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, (long_setup, short_setup, above_ma, below_ma) in enumerate(conditions):
            if position == 1:
                if short_setup:
                    position = 0
            elif position == -1:
                if long_setup:
                    position = 0
            elif long_setup and above_ma:
                position = 1
            elif short_setup and below_ma:
                position = -1
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = InnovationDebtParams()
    index = pd.date_range("2020-01-01", periods=7, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    conditions = pd.DataFrame({
        "long_setup": [True, False, False, False, False, True, False],
        "short_setup": [False, False, True, True, False, False, False],
        "above_ma": [True, True, True, False, False, False, False],
        "below_ma": [False, False, False, True, True, True, True],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, conditions, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0).all()

    t = np.arange(400, dtype=float)
    close = 100 * np.exp(0.0003 * t + 0.03 * np.sin(t / 9))
    history = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000.0,
    }, index=pd.date_range("2020-01-01", periods=len(t), freq="B"))
    full = GeneratedStrategy.indicators(history, params)
    prefix = GeneratedStrategy.indicators(history.iloc[:300], params)
    pd.testing.assert_frame_equal(full.iloc[:300], prefix)
    assert GeneratedStrategy.warmup_bars(params) >= 200
