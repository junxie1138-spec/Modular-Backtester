from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    energy_window: int = 5
    energy_threshold: float = 0.65
    efficiency_window: int = 8
    efficiency_threshold: float = 0.35


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791282113"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(16 + params.energy_window - 1, params.efficiency_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.energy_window < 1 or params.efficiency_window < 1:
            raise ValueError("Window lengths must be positive.")
        if not (0 < params.energy_threshold < 1 and 0 < params.efficiency_threshold <= 1):
            raise ValueError("Thresholds must lie in (0, 1], with energy_threshold below 1.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["log_close"] = np.log(close.where(close > 0))
        total_energy = pd.Series(0.0, index=data.index)
        directional_energy = pd.Series(0.0, index=data.index)
        for half in (1, 2, 4, 8):
            mean = out["log_close"].rolling(half, min_periods=half).mean()
            coefficient = (mean - mean.shift(half)) * np.sqrt(half / 2.0)
            out[f"haar_{half}"] = coefficient
            energy = coefficient.pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
            out[f"energy_{half}"] = energy
            total_energy = total_energy + energy
            if half >= 4:
                signed_energy = (np.sign(coefficient) * coefficient.pow(2)).rolling(
                    params.energy_window, min_periods=params.energy_window
                ).mean()
                out[f"signed_energy_{half}"] = signed_energy
                directional_energy = directional_energy + signed_energy
        out["total_energy"] = total_energy
        out["directional_energy"] = directional_energy
        out["wavelet_score"] = (directional_energy / total_energy.replace(0, np.nan)).fillna(0.0)
        out["log_change"] = out["log_close"].diff()
        out["path_length"] = out["log_change"].abs().rolling(
            params.efficiency_window, min_periods=params.efficiency_window
        ).sum()
        out["displacement"] = out["log_close"].diff(params.efficiency_window)
        out["efficiency"] = (
            out["displacement"] / out["path_length"].replace(0, np.nan)
        ).fillna(0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        bullish = (
            (indicators["wavelet_score"] >= params.energy_threshold)
            & (indicators["efficiency"] >= params.efficiency_threshold)
        ).fillna(False)
        bearish = (
            (indicators["wavelet_score"] <= -params.energy_threshold)
            & (indicators["efficiency"] <= -params.efficiency_threshold)
        ).fillna(False)
        desired = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(desired):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    evidence = pd.DataFrame(
        {"wavelet_score": [0.9, 0.0, -0.9, -0.9, 0.9, 0.9, 0.0],
         "efficiency": [0.8, 0.0, -0.8, -0.8, -0.8, 0.8, 0.0]},
        index=index,
    )
    result = strategy.generate_signals(data, evidence, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    index = pd.date_range("2020-01-01", periods=80)
    prices = 100.0 * np.exp(np.arange(80) * 0.002)
    trend = pd.DataFrame(
        {"open": prices, "high": prices * 1.01, "low": prices * 0.99,
         "close": prices, "volume": 1000.0}, index=index,
    )
    full = strategy.indicators(trend, params)
    prefix = strategy.indicators(trend.iloc[:50], params)
    pd.testing.assert_frame_equal(full.iloc[:50], prefix)
    assert full["wavelet_score"].iloc[-1] > params.energy_threshold
    assert full["efficiency"].iloc[-1] > params.efficiency_threshold
    assert full["energy_8"].first_valid_index() == index[strategy.warmup_bars(params) - 1]
    flat = trend.assign(open=100.0, high=101.0, low=99.0, close=100.0)
    flat_signals = strategy.generate_signals(
        flat, strategy.indicators(flat, params), None, params
    ).data
    assert flat_signals["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
