from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 24
    area_threshold: float = 0.06


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791233878"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return int(params.lookback) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.lookback < 2 or not 0.0 <= params.area_threshold < 0.5:
            raise ValueError("lookback must be >= 2 and area_threshold must be in [0, 0.5)")
        n = params.lookback
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff().replace([np.inf, -np.inf], np.nan)
        out["time"] = np.arange(len(data), dtype=float)
        out["return_moment"] = out["time"] * out["return"]
        out["net_return"] = out["return"].rolling(n, min_periods=n).sum()
        out["path_length"] = out["return"].abs().rolling(n, min_periods=n).sum()
        out["moment_sum"] = out["return_moment"].rolling(n, min_periods=n).sum()
        # Antisymmetric level-two signature of time and cumulative returns:
        # sum((2*k - n - 1)*return_k)/(2*n), normalized by path length.
        out["signed_area"] = (
            2.0 * out["moment_sum"]
            - (2.0 * out["time"] - n + 1.0) * out["net_return"]
        ) / (2.0 * n)
        denominator = out["path_length"].where(out["path_length"] > 0)
        out["area"] = out["signed_area"] / denominator
        out["efficiency"] = out["net_return"] / denominator
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        area = indicators["area"]
        efficiency = indicators["efficiency"]
        valid = np.isfinite(area) & np.isfinite(efficiency)
        candidates = np.select(
            [
                valid & (efficiency >= 0.25) & (area > params.area_threshold),
                valid & (efficiency <= -0.25) & (area < -params.area_threshold),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entered_at = -1
        for i, candidate in enumerate(candidates):
            if position:
                if i - entered_at == 18:
                    position = 0
            elif candidate:
                position = int(candidate)
                entered_at = i
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=80, freq="B")
    returns = np.arange(1, 81, dtype=float) / 10000.0
    data = pd.DataFrame({"close": np.exp(np.cumsum(returns))}, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    n = params.lookback
    weights = 2 * np.arange(1, n + 1) - n - 1
    expected = np.dot(weights, returns[-n:]) / (2 * n * np.abs(returns[-n:]).sum())
    assert np.isclose(derived["area"].iloc[-1], expected)
    assert derived["area"].iloc[:n].isna().all()
    assert GeneratedStrategy.warmup_bars(params) == n + 1
    data = data.iloc[:43]
    direction = np.array([1] * 19 + [-1] * 24)
    synthetic = pd.DataFrame({"area": direction * 0.1, "efficiency": direction * 0.8}, index=data.index)
    signal = GeneratedStrategy.generate_signals(data, synthetic, None, params).data["signal"]
    assert signal.iloc[0] == 0
    assert (signal.iloc[1:19] == 1).all() and signal.iloc[19] == 0
    assert (signal.iloc[20:38] == -1).all() and signal.iloc[38] == 0
