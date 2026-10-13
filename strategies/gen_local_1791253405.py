from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    move_window: int = 4
    volume_window: int = 20
    impact_window: int = 20
    participation: float = 1.1
    pressure_threshold: float = 0.35


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791253405"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        if min(params.move_window, params.volume_window, params.impact_window) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not np.isfinite(params.participation) or params.participation <= 0:
            raise ValueError("participation must be finite and positive")
        if not np.isfinite(params.pressure_threshold) or not 0 < params.pressure_threshold < 1:
            raise ValueError("pressure_threshold must lie between zero and one")
        return int(max(params.volume_window + params.impact_window + 1,
                       params.volume_window + params.move_window))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["absolute_return"] = out["return"].abs()
        out["volume_baseline"] = volume.rolling(params.volume_window).mean().shift(1)
        out["relative_volume"] = (volume / out["volume_baseline"]).replace([np.inf, -np.inf], np.nan)
        out["log_relative_volume"] = np.log(out["relative_volume"])
        out["impact_correlation"] = out["log_relative_volume"].rolling(params.impact_window).corr(out["absolute_return"]).replace([np.inf, -np.inf], np.nan)
        out["lagged_impact"] = out["impact_correlation"].shift(1)
        out["impact_regime"] = np.sign(out["lagged_impact"])
        out["volume_weight"] = out["relative_volume"].clip(upper=4.0)
        out["weighted_return"] = out["return"] * out["volume_weight"]
        out["confirmed_return"] = out["weighted_return"].where(out["relative_volume"] >= params.participation, 0.0)
        out.loc[out["weighted_return"].isna(), "confirmed_return"] = np.nan
        out["pressure_numerator"] = out["confirmed_return"].rolling(params.move_window).sum()
        out["pressure_denominator"] = out["weighted_return"].abs().rolling(params.move_window).sum()
        out["pressure"] = (out["pressure_numerator"] / out["pressure_denominator"].where(out["pressure_denominator"] > 0)).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        self.warmup_bars(params)
        pressure = indicators["pressure"]
        regime = indicators["impact_regime"]
        eligible = np.isfinite(pressure) & np.isfinite(regime) & (pressure.abs() >= params.pressure_threshold)
        candidate = (np.sign(pressure) * regime).where(eligible, 0).fillna(0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, desired in enumerate(candidate.to_numpy()):
            if position == 0:
                position = int(desired)
            elif desired == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = Params()
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    features = pd.DataFrame({"pressure": [1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 1.0, 1.0],
                             "impact_regime": 1.0}, index=index)
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    mirrored = features.copy()
    mirrored["pressure"] *= -1
    inverse = strategy.generate_signals(data, mirrored, None, params).data
    assert inverse["signal"].equals(-result["signal"])
    assert result["size"].gt(0).all()
    features["pressure"] = np.nan
    assert strategy.generate_signals(data, features, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
