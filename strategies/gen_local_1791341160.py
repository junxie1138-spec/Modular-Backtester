from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    volume_window: int = 20
    impact_window: int = 20
    pressure_window: int = 4
    pressure_threshold: float = 0.25
    volume_confirmation: float = 1.10
    congestion_threshold: float = 0.75


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791341160"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.volume_window + params.impact_window + 2 * params.pressure_window + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.volume_window, params.impact_window, params.pressure_window) < 2:
            raise ValueError("Windows must be at least two bars.")
        if not (0 < params.pressure_threshold < 1 and params.volume_confirmation > 0 and 0 < params.congestion_threshold < 1):
            raise ValueError("Invalid signal thresholds.")
        out = pd.DataFrame(index=data.index)
        volume = data["volume"].clip(lower=0)
        out["volume_baseline"] = volume.rolling(params.volume_window).mean().shift(1)
        out["relative_volume"] = volume.div(out["volume_baseline"].replace(0, np.nan))
        out["return"] = data["close"].pct_change(fill_method=None)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["body"] = ((data["close"] - data["open"]) / out["true_range"].replace(0, np.nan)).clip(-1, 1)
        out["weighted_body"] = out["body"] * out["relative_volume"]
        out["pressure"] = out["weighted_body"].rolling(params.pressure_window).sum().div(
            out["relative_volume"].rolling(params.pressure_window).sum().replace(0, np.nan)
        )
        out["volume_confirmation"] = out["relative_volume"].rolling(params.pressure_window).mean()
        out["impact"] = out["return"].abs().div(out["relative_volume"].replace(0, np.nan))
        out["recent_impact"] = out["impact"].rolling(params.pressure_window).mean()
        out["historical_impact"] = out["impact"].rolling(params.impact_window).mean().shift(params.pressure_window)
        out["impact_ratio"] = out["recent_impact"].div(out["historical_impact"].replace(0, np.nan))
        out = out.replace([np.inf, -np.inf], np.nan)
        out["congested"] = (out["impact_ratio"] < params.congestion_threshold).astype(float)
        out["strength"] = out["pressure"].abs().clip(0, 1)
        ready = out[["pressure", "volume_confirmation", "impact_ratio"]].notna().all(axis=1)
        ready &= np.arange(len(data)) >= self.warmup_bars(params) - 1
        confirmed = ready & (out["volume_confirmation"] >= params.volume_confirmation)
        direction = np.sign(out["pressure"].fillna(0))
        regime_direction = direction.where(out["congested"] == 0, -direction)
        out["entry"] = regime_direction.where(confirmed & (out["strength"] >= params.pressure_threshold), 0).astype(int)
        return out.fillna(0.0)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        position = 0
        position_size = 0.20
        for i, (entry, strength) in enumerate(zip(indicators["entry"].to_numpy(), indicators["strength"].to_numpy())):
            entry = int(entry)
            if position and entry == -position:
                position = 0
                position_size = 0.20
            elif position == 0 and entry:
                position = entry
                position_size = float(np.clip(0.20 + 0.75 * strength, 0.20, 0.95))
            elif position and entry == position:
                position_size = float(np.clip(0.20 + 0.75 * strength, 0.20, 0.95))
            signals[i] = position
            sizes[i] = position_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=7))
    derived = pd.DataFrame({
        "entry": [1, 0, 1, -1, -1, 0, 1],
        "strength": [0.4, 0.0, 0.8, 0.6, 0.6, 0.0, 0.5],
    }, index=data.index)
    result = strategy.generate_signals(data, derived, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1]
    assert np.isclose(result["size"].iloc[1], 0.50)
    assert np.isclose(result["size"].iloc[3], 0.80)
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
