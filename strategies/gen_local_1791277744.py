from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 8
    percentile_bars: int = 126
    area_percentile: float = 0.85
    volatility_bars: int = 20
    recovery_bars: int = 3
    holding_bars: int = 8
    target_volatility: float = 0.12
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791277744"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return max(params.path_bars + params.percentile_bars + 1,
                   params.volatility_bars + 1, params.recovery_bars + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if min(params.path_bars, params.percentile_bars, params.volatility_bars,
               params.recovery_bars, params.holding_bars) < 1:
            raise ValueError("Lookbacks and holding_bars must be positive.")
        if not (0.0 < params.area_percentile < 1.0
                and 0.0 < params.target_volatility
                and 0.0 < params.max_size <= 1.0):
            raise ValueError("Invalid percentile or sizing parameters.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff()
        valid_range = (data["low"] > 0) & (data["high"] >= data["low"])
        out["log_range"] = np.log(
            (data["high"] / data["low"]).where(valid_range)
        )
        out["range_change"] = out["log_range"].diff()
        x = pd.Series(0.0, index=data.index)
        y = pd.Series(0.0, index=data.index)
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature of return and range-change increments.
        for lag in range(params.path_bars - 1, -1, -1):
            dx = out["return"].shift(lag)
            dy = out["range_change"].shift(lag)
            area = area + 0.5 * (y * dx - x * dy)
            x = x + dx
            y = y + dy
        out["area"] = area
        out["path_return"] = x
        out["path_range_change"] = y
        out["area_threshold"] = out["area"].rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.area_percentile).shift(1)
        out["recovery"] = out["return"].rolling(
            params.recovery_bars, min_periods=params.recovery_bars
        ).sum()
        out["annual_volatility"] = out["return"].rolling(
            params.volatility_bars, min_periods=params.volatility_bars
        ).std(ddof=0) * np.sqrt(252.0)
        out["target_size"] = (
            params.target_volatility / out["annual_volatility"].where(
                out["annual_volatility"] > 0
            )
        ).clip(upper=params.max_size)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            out[["area", "area_threshold", "path_return", "path_range_change",
                 "recovery", "target_size"]].notna().all(axis=1)
            & (out["area"] > out["area_threshold"])
            & (out["area"] > 0)
            & (out["path_return"] < 0)
            & (out["path_range_change"] < 0)
            & (out["recovery"] > 0)
            & (out["target_size"] > 0)
        )
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive.")
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        proposed_size = indicators["target_size"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), params.max_size, dtype=float)
        exit_bar = -1
        held_size = params.max_size
        for i in range(len(data)):
            if i == exit_bar:
                # Reserve this bar for the fixed exit, even if entry remains true.
                continue
            if i > exit_bar and entry[i]:
                candidate = proposed_size[i]
                if not np.isfinite(candidate) or candidate <= 0:
                    continue
                held_size = min(candidate, params.max_size)
                exit_bar = i + params.holding_bars
            if i < exit_bar:
                signal[i] = 1
                size[i] = held_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = SignatureParams(holding_bars=2)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"entry": True, "target_size": np.arange(1, 9) / 10.0}, index=index
    )
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0, 1]
    assert np.allclose(result.loc[index[[1, 2, 4, 5, 7]], "size"],
                       [0.1, 0.1, 0.4, 0.4, 0.7])
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
