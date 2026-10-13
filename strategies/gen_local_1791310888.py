from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RegimeParams:
    ma_window: int = 5
    regime_window: int = 24
    horizon: int = 4
    threshold: float = 0.6


class GeneratedStrategy(BaseStrategy[RegimeParams]):
    strategy_id = "gen_local_1791310888"

    @classmethod
    def params_type(cls):
        return RegimeParams

    def warmup_bars(self, params: RegimeParams) -> int:
        return int(max(params.ma_window, params.regime_window + params.horizon))

    def indicators(self, data: pd.DataFrame, params: RegimeParams) -> pd.DataFrame:
        if min(params.ma_window, params.regime_window) < 2 or params.horizon < 1 or not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("Windows must be >= 2, horizon >= 1, and threshold finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["price_std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["price_std"].replace(0.0, np.nan)
        out["log_close"] = np.log(close.where(close > 0))
        out["daily_return"] = out["log_close"].diff()
        out["horizon_return"] = out["log_close"].diff(params.horizon)
        out["daily_variance"] = out["daily_return"].rolling(params.regime_window, min_periods=params.regime_window).var(ddof=0)
        out["horizon_variance"] = out["horizon_return"].rolling(params.regime_window, min_periods=params.regime_window).var(ddof=0)
        out["variance_ratio"] = out["horizon_variance"] / (params.horizon * out["daily_variance"].replace(0.0, np.nan))
        out["orientation"] = np.where(out["variance_ratio"] > 1.0, 1.0, -1.0)
        out["orientation"] = out["orientation"].where(out["variance_ratio"].notna())
        out["score"] = out["orientation"] * out["z"]
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _positions(scores: np.ndarray, threshold: float) -> np.ndarray:
        result = np.zeros(len(scores), dtype=int)
        held = 0
        for i, score in enumerate(scores):
            if np.isfinite(score):
                if score >= threshold:
                    held = 1
                elif score <= -threshold:
                    held = 0
            result[i] = held
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RegimeParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = self._positions(indicators["score"].to_numpy(dtype=float), params.threshold)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    scores = np.array([np.nan, 0.6, 0.1, np.nan, -0.6, -0.1, 0.6])
    assert GeneratedStrategy._positions(scores, 0.6).tolist() == [0, 1, 1, 1, 0, 0, 1]
