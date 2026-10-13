from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AsymmetricEchoParams:
    estimation_bars: int = 63
    percentile_bars: int = 126
    entry_percentile: float = 0.85
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[AsymmetricEchoParams]):
    strategy_id = "gen_local_1791245211"

    @classmethod
    def params_type(cls) -> type[AsymmetricEchoParams]:
        return AsymmetricEchoParams

    @staticmethod
    def warmup_bars(params: AsymmetricEchoParams) -> int:
        return int(params.estimation_bars + params.percentile_bars + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: AsymmetricEchoParams) -> pd.DataFrame:
        if params.estimation_bars < 2 or params.percentile_bars < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie between zero and one")
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")

        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lag_return"] = out["return"].shift(1)
        valid = out["return"].notna() & out["lag_return"].notna()
        for name, mask in (("up", out["lag_return"] > 0), ("down", out["lag_return"] < 0)):
            out[name + "_product"] = (out["return"] * out["lag_return"]).where(mask, 0.0).where(valid)
            out[name + "_energy"] = out["lag_return"].pow(2).where(mask, 0.0).where(valid)
            out[name + "_numerator"] = out[name + "_product"].rolling(params.estimation_bars).sum()
            out[name + "_denominator"] = out[name + "_energy"].rolling(params.estimation_bars).sum()
            out[name + "_beta"] = out[name + "_numerator"] / out[name + "_denominator"].replace(0.0, np.nan)
        out["selected_beta"] = out["up_beta"].where(out["return"] > 0, out["down_beta"])
        out["forecast"] = (out["return"] * out["selected_beta"]).replace([np.inf, -np.inf], np.nan)
        out["forecast_magnitude"] = out["forecast"].abs()
        out["historical_magnitude"] = out["forecast_magnitude"].shift(1)
        out["entry_threshold"] = out["historical_magnitude"].rolling(params.percentile_bars).quantile(params.entry_percentile)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AsymmetricEchoParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")
        forecast = indicators["forecast"].to_numpy(dtype=float)
        threshold = indicators["entry_threshold"].to_numpy(dtype=float)
        eligible = np.isfinite(forecast) & np.isfinite(threshold) & (np.abs(forecast) > threshold)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i == exit_bar:
                    position = 0
                signals[i] = position
                continue
            if eligible[i]:
                position = 1 if forecast[i] > 0 else -1
                exit_bar = i + params.holding_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = AsymmetricEchoParams()
    index = pd.date_range("2020-01-01", periods=42, freq="B")
    data = pd.DataFrame(index=index)
    features = pd.DataFrame({"forecast": np.where(np.arange(42) < 2, 1.0, -1.0), "entry_threshold": 0.5}, index=index)
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19] == 0
    assert result["signal"].iloc[20:38].eq(-1).all()
    assert result["signal"].iloc[38] == 0
    assert result["size"].gt(0).all()
