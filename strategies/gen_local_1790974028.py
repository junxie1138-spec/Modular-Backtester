from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRegimeParams:
    regime_window: int = 20
    pressure_window: int = 8
    regime_threshold: float = 0.15
    entry_threshold: float = 0.30


class GeneratedStrategy(BaseStrategy[GapRegimeParams]):
    strategy_id = "gen_local_1790974028"

    @classmethod
    def params_type(cls):
        return GapRegimeParams

    def warmup_bars(self, params: GapRegimeParams) -> int:
        return int(max(params.regime_window, params.pressure_window) + 1)

    def indicators(self, data: pd.DataFrame, params: GapRegimeParams) -> pd.DataFrame:
        if min(params.regime_window, params.pressure_window) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not (0 < params.regime_threshold <= 1 and 0 < params.entry_threshold <= 1):
            raise ValueError("Thresholds must be in (0, 1].")
        out = pd.DataFrame(index=data.index)
        out["gap"] = data["open"].div(data["close"].shift(1).replace(0, np.nan)).sub(1)
        out["body"] = data["close"].div(data["open"].replace(0, np.nan)).sub(1)
        out[["gap", "body"]] = out[["gap", "body"]].replace([np.inf, -np.inf], np.nan)
        out["gap_body_sum"] = (out["gap"] * out["body"]).rolling(params.regime_window).sum()
        out["gap_energy"] = out["gap"].pow(2).rolling(params.regime_window).sum()
        out["body_energy"] = out["body"].pow(2).rolling(params.regime_window).sum()
        out["alignment"] = out["gap_body_sum"].div(
            np.sqrt(out["gap_energy"] * out["body_energy"]).replace(0, np.nan)
        ).clip(-1, 1)
        out["gap_sum"] = out["gap"].rolling(params.pressure_window).sum()
        out["absolute_gap_sum"] = out["gap"].abs().rolling(params.pressure_window).sum()
        out["pressure"] = out["gap_sum"].div(out["absolute_gap_sum"].replace(0, np.nan)).clip(-1, 1)
        out["regime"] = pd.Series(
            np.select(
                [out["alignment"] >= params.regime_threshold,
                 out["alignment"] <= -params.regime_threshold],
                [1, -1],
                default=np.nan,
            ),
            index=data.index,
        ).ffill().fillna(0).astype(int)
        out["score"] = (out["regime"] * out["pressure"]).where(out["alignment"].notna())
        out["conviction"] = (
            out["score"].abs() * (0.5 + 0.5 * out["alignment"].abs())
        ).clip(0, 1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRegimeParams,
    ) -> SignalFrame:
        state = 0
        held_size = 0.20
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        scores = indicators["score"].to_numpy(dtype=float)
        convictions = indicators["conviction"].to_numpy(dtype=float)
        for i in range(len(data)):
            score = scores[i]
            desired = 0
            if np.isfinite(score):
                if score >= params.entry_threshold:
                    desired = 1
                elif score <= -params.entry_threshold:
                    desired = -1
            if state and desired == -state:
                state = 0
                held_size = 0.20
            elif state == 0 and desired and np.isfinite(convictions[i]):
                state = desired
                held_size = float(np.clip(0.20 + 0.75 * convictions[i], 0.20, 0.95))
            signals[i] = state
            sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapRegimeParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame(
        {"score": [np.nan, 0.6, 0.1, -0.6, -0.7, -0.1, 0.6, 0.7],
         "conviction": [np.nan, 0.6, 0.1, 0.6, 0.7, 0.1, 0.6, 0.7]},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert np.isclose(result["size"].iloc[2], 0.65)
    assert result["size"].iloc[3] == result["size"].iloc[2]
    assert np.isclose(result["size"].iloc[5], 0.725)
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
