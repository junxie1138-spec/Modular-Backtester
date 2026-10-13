from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationParams:
    shock_z: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[InnovationParams]):
    strategy_id = "gen_local_1791214887"

    @classmethod
    def params_type(cls) -> type[InnovationParams]:
        return InnovationParams

    def warmup_bars(self, params: InnovationParams) -> int:
        return 62

    def indicators(self, data: pd.DataFrame, params: InnovationParams) -> pd.DataFrame:
        if not np.isfinite(params.shock_z) or params.shock_z <= 0:
            raise ValueError("shock_z must be finite and positive")
        if type(params.hold_bars) is not int or not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be an integer between 3 and 5")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(np.isfinite(close) & (close > 0)))
        out["time_index"] = np.arange(len(data), dtype=float)
        out["return"] = out["log_close"].diff()
        out["mean_log"] = out["log_close"].rolling(20, min_periods=20).mean()
        # OLS slope: covariance with bar number divided by its fixed window variance.
        out["slope"] = out["log_close"].rolling(20, min_periods=20).cov(
            out["time_index"], ddof=0
        ) / ((20 ** 2 - 1) / 12.0)
        out["endpoint"] = out["mean_log"] + 9.5 * out["slope"]
        out["forecast"] = out["endpoint"].shift(1) + out["slope"].shift(1)
        out["surprise"] = out["log_close"] - out["forecast"]
        out["slope_innovation"] = out["slope"].diff()
        out["innovation_mean"] = out["slope_innovation"].rolling(40, min_periods=40).mean().shift(1)
        out["innovation_noise"] = out["slope_innovation"].rolling(40, min_periods=40).std(ddof=0).shift(1)
        out["shock_score"] = (
            out["slope_innovation"] - out["innovation_mean"]
        ) / out["innovation_noise"].where(out["innovation_noise"] > 1e-12)
        out["return_noise"] = out["return"].rolling(40, min_periods=40).std(ddof=0).shift(1)
        out["drift_ratio"] = out["slope"].shift(1).abs() / out["return_noise"].where(out["return_noise"] > 1e-12)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: InnovationParams,
    ) -> SignalFrame:
        if type(params.hold_bars) is not int or not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be an integer between 3 and 5")
        if not np.isfinite(params.shock_z) or params.shock_z <= 0:
            raise ValueError("shock_z must be finite and positive")
        valid = np.isfinite(indicators[["shock_score", "surprise", "drift_ratio"]]).all(axis=1)
        quiet = valid & (indicators["drift_ratio"] <= 0.25)
        long_entry = quiet & (indicators["shock_score"] <= -params.shock_z) & (indicators["surprise"] < 0)
        short_entry = quiet & (indicators["shock_score"] >= params.shock_z) & (indicators["surprise"] > 0)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i, candidate in enumerate(candidates):
            if position != 0 and i == exit_at:
                position = 0
            elif position == 0 and candidate != 0:
                position = int(candidate)
                exit_at = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    for hold in (3, 4, 5):
        index = pd.date_range("2020-01-01", periods=2 * hold + 4)
        data = pd.DataFrame(index=index)
        ind = pd.DataFrame({"shock_score": 3.0, "surprise": 1.0, "drift_ratio": 0.0}, index=index)
        ind.iloc[0] = [-3.0, -1.0, 0.0]
        result = strategy.generate_signals(data, ind, None, InnovationParams(hold_bars=hold)).data
        signal = result["signal"].tolist()
        assert signal[:hold + 2] == [0] + [1] * hold + [0]
        assert signal[hold + 2:2 * hold + 2] == [-1] * hold
        assert signal[2 * hold + 2] == 0
        assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
