from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureSeasonParams:
    path_bars: int = 4
    area_threshold: float = 0.025


class GeneratedStrategy(BaseStrategy[SignatureSeasonParams]):
    strategy_id = "gen_local-w2_1791278397"

    @classmethod
    def params_type(cls):
        return SignatureSeasonParams

    @staticmethod
    def warmup_bars(params: SignatureSeasonParams) -> int:
        return max(2, int(params.path_bars)) + 1

    def indicators(self, data: pd.DataFrame, params: SignatureSeasonParams) -> pd.DataFrame:
        window = max(2, int(params.path_bars))
        out = pd.DataFrame(index=data.index)
        out["phase"] = 4.0 * np.pi * (data.index.day - 1) / data.index.days_in_month
        out["clock"] = np.sin(out["phase"])
        out["log_price"] = np.log(data["close"].where(data["close"] > 0))
        out["dx"] = out["log_price"].diff()
        out["dy"] = out["clock"].diff()
        x = pd.Series(0.0, index=data.index)
        y = pd.Series(0.0, index=data.index)
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature of the translated local path.
        for lag in range(window - 1, -1, -1):
            step_x = out["dx"].shift(lag)
            step_y = out["dy"].shift(lag)
            area = area + 0.5 * (x * step_y - y * step_x)
            x = x + step_x
            y = y + step_y
        out["path_x"] = x
        out["path_y"] = y
        out["area"] = area
        out["price_variation"] = out["dx"].abs().rolling(window, min_periods=window).sum()
        out["clock_variation"] = out["dy"].abs().rolling(window, min_periods=window).sum()
        out["area_scale"] = out["price_variation"] * out["clock_variation"]
        out["normalized_area"] = out["area"] / out["area_scale"].where(out["area_scale"] > 0)
        # Positive area matches price ~ cos(phase), whose tangent is -sin(phase).
        out["score"] = -out["normalized_area"] * out["clock"]
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureSeasonParams) -> SignalFrame:
        threshold = max(float(params.area_threshold), 1e-12)
        score = indicators["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        directions = np.where(score >= threshold, 1, np.where(score <= -threshold, -1, 0))
        positions = np.zeros(len(data), dtype=int)
        position = 0
        previous = 0
        for i, value in enumerate(directions):
            direction = int(value)
            if position:
                if direction == -position:
                    position = 0
            elif direction != 0 and direction == previous:
                position = direction
            positions[i] = position
            previous = direction
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = SignatureSeasonParams()
    index = pd.bdate_range("2020-01-01", periods=9)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"score": [1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, -1, 0]
    assert result.data["size"].gt(0).all()
    index = pd.bdate_range("2020-01-01", periods=80)
    phase = 4.0 * np.pi * (index.day - 1) / index.days_in_month
    close = 100.0 * np.exp(0.01 * np.cos(phase))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert np.isfinite(derived.to_numpy()).all()
    assert derived["score"].iloc[:params.path_bars].eq(0).all()
    cutoff = 40
    pd.testing.assert_frame_equal(derived.iloc[:cutoff], strategy.indicators(data.iloc[:cutoff], params))
    assert strategy.warmup_bars(params) >= params.path_bars + 1


if __name__ == "__main__":
    _self_check()
