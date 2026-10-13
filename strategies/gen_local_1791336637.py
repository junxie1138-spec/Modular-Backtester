from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapContractionParams:
    noise_window: int = 20
    atr_window: int = 14
    gap_threshold: float = 0.35
    contraction_ratio: float = 0.60
    stop_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[GapContractionParams]):
    strategy_id = "gen_local_1791336637"

    @classmethod
    def params_type(cls) -> type[GapContractionParams]:
        return GapContractionParams

    def warmup_bars(self, params: GapContractionParams) -> int:
        return int(max(params.noise_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapContractionParams) -> pd.DataFrame:
        if params.noise_window < 2 or params.atr_window < 1:
            raise ValueError("Lookback windows must be positive; noise_window must be at least two.")
        if not (params.gap_threshold > 0 and 0 < params.contraction_ratio < 1 and params.stop_atr > 0):
            raise ValueError("Invalid gap, contraction, or stop parameters.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        denominator = out["previous_close"].where(out["previous_close"] > 0)
        out["return"] = (data["close"] / denominator - 1.0).replace([np.inf, -np.inf], np.nan)
        out["previous_return"] = out["return"].shift(1)
        out["noise"] = out["return"].rolling(params.noise_window, min_periods=params.noise_window).std(ddof=0).shift(1)
        out["gap"] = (data["open"] / denominator - 1.0).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["gap_agrees"] = (out["noise"] > 0) & (out["gap"] > params.gap_threshold * out["noise"])
        out["return_agrees"] = (
            (out["previous_return"] < -0.5 * out["noise"])
            & (out["return"] < 0)
            & (out["return"] > params.contraction_ratio * out["previous_return"])
        )
        out["entry"] = (
            out["gap_agrees"] & out["return_agrees"]
            & out["atr"].gt(0) & np.isfinite(out["atr"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapContractionParams) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=np.int64)
        pending = False
        active = False
        held_sessions = 0
        fixed_stop = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    # Entry fills at this open; freeze the ATR known at the decision close.
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
                    active = True
                    held_sessions = 0
            if active:
                held_sessions += 1
                # Close-based exits fill next bar, including the two-session expiry.
                if not np.isfinite(closes[i]) or closes[i] < fixed_stop or held_sessions >= 2:
                    active = False
                    decisions[i] = 0
                else:
                    decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 96.0,
        "close": [100.0, 97.0, 100.0, 100.0, 101.0, 101.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, True, False, False, False],
        "atr": [1.0, 20.0, 1.0, 1.0, 20.0, 20.0, 1.0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, GapContractionParams()).data
    # First trade stops using frozen entry ATR; second expires after two sessions.
    assert result["signal"].tolist() == [0, 1, 0, 0, 1, 1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
