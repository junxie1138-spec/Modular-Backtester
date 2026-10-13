from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapInnovationParams:
    level_span: int = 8
    slope_span: int = 4
    volatility_window: int = 20
    min_innovation: float = 0.02
    min_gap_sigma: float = 0.35
    min_repair: float = 0.40
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapInnovationParams]):
    strategy_id = "gen_local-w2_1791202605"

    @classmethod
    def params_type(cls) -> type[GapInnovationParams]:
        return GapInnovationParams

    def warmup_bars(self, params: GapInnovationParams) -> int:
        return int(max(max(2, params.volatility_window) + 1,
                       max(2, params.level_span) + max(2, params.slope_span) + 1))

    def indicators(self, data: pd.DataFrame, params: GapInnovationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["log_close"] = np.log(close)
        out["return"] = out["log_close"].diff()
        window = max(2, params.volatility_window)
        out["volatility"] = out["return"].rolling(window, min_periods=window).std(ddof=0)
        level_span = max(2, params.level_span)
        slope_span = max(2, params.slope_span)
        out["latent_level"] = out["log_close"].ewm(
            span=level_span, adjust=False, min_periods=level_span
        ).mean()
        out["level_change"] = out["latent_level"].diff()
        out["latent_slope"] = out["level_change"].ewm(
            span=slope_span, adjust=False, min_periods=slope_span
        ).mean()
        scale = out["volatility"].where(out["volatility"] > 0)
        out["slope_innovation"] = out["latent_slope"].diff() / scale
        out["previous_close"] = close.shift(1)
        out["gap"] = np.log(opening / out["previous_close"])
        out["gap_sigma"] = out["gap"] / scale
        displacement = out["previous_close"] - opening
        out["repair_fraction"] = (close - opening) / displacement.where(displacement > 0)
        out["trend_agrees"] = out["slope_innovation"] > params.min_innovation
        out["gap_agrees"] = (
            (out["gap_sigma"] <= -params.min_gap_sigma)
            & (out["repair_fraction"] >= params.min_repair)
            & (out["repair_fraction"] < 1.0)
        )
        out["entry"] = (out["trend_agrees"] & out["gap_agrees"]).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapInnovationParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        hold = max(1, min(2, int(params.holding_bars)))
        signals = np.zeros(len(data), dtype=int)
        exit_bar = -1
        active = False
        for i, entry in enumerate(entries):
            if active:
                if i == exit_bar:
                    active = False
                else:
                    signals[i] = 1
            elif entry:
                active = True
                exit_bar = i + hold
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": True}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, GapInnovationParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0]
    assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
