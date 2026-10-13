from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAcceptanceParams:
    channel_bars: int = 8
    noise_bars: int = 20
    min_gap_atr: float = 0.35


class GeneratedStrategy(BaseStrategy[GapAcceptanceParams]):
    strategy_id = "gen_local-w2_1791282079"

    @classmethod
    def params_type(cls) -> type[GapAcceptanceParams]:
        return GapAcceptanceParams

    @staticmethod
    def warmup_bars(params: GapAcceptanceParams) -> int:
        return max(params.channel_bars, params.noise_bars) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapAcceptanceParams) -> pd.DataFrame:
        if params.channel_bars < 1 or params.noise_bars < 1 or not np.isfinite(params.min_gap_atr) or params.min_gap_atr < 0:
            raise ValueError("Lookbacks must be positive and min_gap_atr finite and nonnegative.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["prior_ceiling"] = data["high"].shift(1).rolling(params.channel_bars, min_periods=params.channel_bars).max()
        out["gap"] = data["open"] - out["prior_close"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["prior_atr"] = out["true_range"].shift(1).rolling(params.noise_bars, min_periods=params.noise_bars).mean()
        out["gap_score"] = out["gap"] / out["prior_atr"].where(out["prior_atr"] > 0)
        out["gap_midpoint"] = (data["open"] + out["prior_close"]) / 2.0
        out["candidate_anchor"] = out[["prior_ceiling", "gap_midpoint"]].max(axis=1, skipna=False)
        out["breakout"] = (
            (data["open"] > out["prior_ceiling"])
            & (out["gap"] > 0)
            & (out["gap_score"] >= params.min_gap_atr)
            & np.isfinite(out["candidate_anchor"])
            & np.isfinite(out["gap_score"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAcceptanceParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        anchor = np.nan
        confirmations = 0
        holding = False
        closes = data["close"].to_numpy(dtype=float)
        events = indicators["breakout"].fillna(False).to_numpy(dtype=bool)
        anchors = indicators["candidate_anchor"].to_numpy(dtype=float)
        for i, close in enumerate(closes):
            if not np.isfinite(close):
                signals[i] = int(holding)
                if not holding:
                    anchor = np.nan
                    confirmations = 0
                continue
            if np.isfinite(anchor):
                if close <= anchor:
                    holding = False
                    anchor = np.nan
                    confirmations = 0
                    continue
                confirmations += 1
                if confirmations >= 2:
                    holding = True
            elif events[i] and np.isfinite(anchors[i]) and close > anchors[i]:
                anchor = anchors[i]
                confirmations = 1
            signals[i] = int(holding)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [101.0, 102.0, 103.0, 99.0, 104.0, 105.0]}, index=index)
    features = pd.DataFrame({"breakout": [True, False, False, False, False, False], "candidate_anchor": 100.0}, index=index)
    result = GeneratedStrategy.generate_signals(data, features, None, GapAcceptanceParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 99.0
    result = GeneratedStrategy.generate_signals(data, features, None, GapAcceptanceParams()).data
    assert result["signal"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
