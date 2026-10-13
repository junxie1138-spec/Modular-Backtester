from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    change_threshold: float = 1.5
    trail_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = "gen_local-w2_1791227736"

    @classmethod
    def params_type(cls):
        return ParticipationParams

    def warmup_bars(self, params: ParticipationParams) -> int:
        # Includes the volume baseline, participation window, and displacement.
        return 44

    def indicators(self, data: pd.DataFrame, params: ParticipationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["close_change"] = data["close"].diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["volume_base"] = data["volume"].rolling(20, min_periods=20).mean().shift(1)
        out["relative_volume"] = (
            data["volume"] / out["volume_base"].replace(0, np.nan)
        ).clip(0, 4)
        out["participation"] = (
            out["close_change"] / out["previous_atr"].replace(0, np.nan)
            * out["relative_volume"]
        )
        out["recent"] = out["participation"].rolling(3, min_periods=3).mean()
        out["baseline"] = out["participation"].rolling(20, min_periods=20).mean().shift(3)
        out["dispersion"] = (
            out["participation"].rolling(20, min_periods=20)
            .std(ddof=0).shift(3).clip(lower=0.25)
        )
        out["change"] = (out["recent"] - out["baseline"]) / out["dispersion"]
        out["confirmation"] = out["relative_volume"].rolling(3, min_periods=3).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["ready"] = (
            out[["change", "atr", "confirmation"]].notna().all(axis=1)
            & out["atr"].gt(0)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ParticipationParams,
    ) -> SignalFrame:
        if (
            not np.isfinite(params.change_threshold) or params.change_threshold <= 0
            or not np.isfinite(params.trail_atr) or params.trail_atr <= 0
        ):
            raise ValueError("Parameters must be finite and positive")
        signals = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        armed, activated = True, False
        entry, stop, entry_atr = np.nan, -np.inf, np.nan
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            ready = bool(indicators["ready"].iloc[i])
            change = float(indicators["change"].iloc[i])
            if ready and abs(change) <= params.change_threshold / 2.0:
                armed = True
            if side:
                # The preceding decision fills at this session's open.
                if age == 0:
                    entry = float(data["open"].iloc[i])
                    stop = side * entry - 2.0 * entry_atr
                age += 1
                if not np.isfinite(close) or not np.isfinite(entry) or entry <= 0:
                    side = 0
                    continue
                # Stops are evaluated at the close; exit orders fill next bar.
                # Signed-price stops only rise: short price stops only fall.
                favorable = side * close
                if favorable <= stop or age >= 4:
                    side = 0
                    continue
                if side * (close / entry - 1.0) >= 0.01:
                    activated = True
                if activated:
                    stop = max(stop, side * entry)
                    atr = float(indicators["atr"].iloc[i])
                    if np.isfinite(atr) and atr > 0:
                        stop = max(stop, favorable - params.trail_atr * atr)
                signals[i] = side
                continue
            if not ready or not np.isfinite(close) or close <= 0:
                continue
            if (
                armed and abs(change) >= params.change_threshold
                and float(indicators["confirmation"].iloc[i]) >= 1.15
            ):
                side = 1 if change > 0 else -1
                age, activated, armed = 0, False, False
                entry_atr = float(indicators["atr"].iloc[i])
                signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({
        "open": 100.0,
        "close": [100, 102, 99, 100, 98, 101, 100, 100, 100, 100, 100, 100],
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 1.0,
        "change": [2, 0, 0, -2, 0, 0, 2, 0, 0, 0, 0, 0],
        "confirmation": 2.0,
        "ready": True,
    }, index=index)
    result = GeneratedStrategy().generate_signals(
        data, indicators, StrategyContext("SPY", "1d", 44), ParticipationParams()
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1, 1, 1, 1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
