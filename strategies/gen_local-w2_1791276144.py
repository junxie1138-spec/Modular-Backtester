from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    memory: int = 40
    entry_score: float = 0.6


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791276144"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(4, int(params.memory)) + 14

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        window = max(4, int(params.memory))
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        ma = close.rolling(10, min_periods=10).mean()
        scale = close.rolling(10, min_periods=10).std(ddof=0)
        z = (close - ma) / scale.where(scale > 1e-12)
        x1, x2 = z.shift(1), z.shift(2)
        # Use identical complete observations for all regression moments.
        complete = z.notna() & x1.notna() & x2.notna()
        y, x1, x2 = z.where(complete), x1.where(complete), x2.where(complete)
        my = y.rolling(window, min_periods=window).mean()
        m1 = x1.rolling(window, min_periods=window).mean()
        m2 = x2.rolling(window, min_periods=window).mean()
        v1 = x1.rolling(window, min_periods=window).var(ddof=0)
        v2 = x2.rolling(window, min_periods=window).var(ddof=0)
        c12 = (x1 * x2).rolling(window, min_periods=window).mean() - m1 * m2
        c1y = (x1 * y).rolling(window, min_periods=window).mean() - m1 * my
        c2y = (x2 * y).rolling(window, min_periods=window).mean() - m2 * my
        ridge = 0.05 * (v1 + v2) + 1e-12
        a, b = v1 + ridge, v2 + ridge
        determinant = (a * b - c12 * c12).where(lambda s: s > 1e-20)
        beta1 = (b * c1y - c12 * c2y) / determinant
        beta2 = (a * c2y - c12 * c1y) / determinant
        intercept = my - beta1 * m1 - beta2 * m2
        previous, current = z.shift(1), z
        for _ in range(4):
            forecast = intercept + beta1 * current + beta2 * previous
            previous, current = current, forecast
        displacement_scale = z.diff(4).rolling(window, min_periods=window).std(ddof=0)
        score = (current - z) / displacement_scale.where(displacement_scale > 1e-12)
        return pd.DataFrame({
            "ma": ma,
            "scale": scale,
            "z": z,
            "lag1": x1,
            "lag2": x2,
            "mean_y": my,
            "mean_lag1": m1,
            "mean_lag2": m2,
            "variance_lag1": v1,
            "variance_lag2": v2,
            "covariance_lags": c12,
            "covariance_lag1_y": c1y,
            "covariance_lag2_y": c2y,
            "ridge": ridge,
            "determinant": determinant,
            "beta1": beta1,
            "beta2": beta2,
            "intercept": intercept,
            "forecast_z": current,
            "displacement_scale": displacement_scale,
            "score": score,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        threshold = max(0.0, float(params.entry_score))
        score, z = indicators["score"], indicators["z"]
        valid = np.isfinite(score) & np.isfinite(z) & z.abs().ge(0.5)
        entry = pd.Series(0, index=data.index, dtype=int)
        entry.loc[valid & score.gt(threshold)] = 1
        entry.loc[valid & score.lt(-threshold)] = -1
        entry.iloc[:GeneratedStrategy.warmup_bars(params)] = 0
        state = 0
        signals = np.zeros(len(data), dtype=int)
        for i, direction in enumerate(entry.to_numpy()):
            if state == 0:
                state = int(direction)
            elif direction == -state:
                # Opposite qualified entry evidence closes the position.
                # A subsequent qualifying bar may open the opposite side.
                state = 0
            signals[i] = state
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = Params(memory=4)
    index = pd.date_range("2020-01-01", periods=24)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    evidence = pd.DataFrame({"z": 1.0, "score": 0.0}, index=index)
    evidence.iloc[18:, evidence.columns.get_loc("score")] = [1, 0, -1, -1, 0, 1]
    result = GeneratedStrategy.generate_signals(data, evidence, None, params).data
    assert result["signal"].tolist() == [0] * 19 + [1, 1, 0, -1, -1]
    assert result["size"].gt(0).all()
    flat = GeneratedStrategy.indicators(data, params)
    assert GeneratedStrategy.generate_signals(data, flat, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
