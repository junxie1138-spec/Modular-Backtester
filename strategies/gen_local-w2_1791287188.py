from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    innovation_threshold: float = 0.75
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791287188"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # Twelve price observations, a slope difference, a lag, and forty pairs.
        return 54

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold <= 0:
            raise ValueError("innovation_threshold must be finite and positive")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(np.isfinite(close) & (close > 0)))
        # Ordinary least-squares slope over twelve equally spaced observations.
        out["slope"] = sum(
            (j - 5.5) * out["log_close"].shift(11 - j)
            for j in range(12)
        ) / 143.0
        out["innovation"] = out["slope"].diff()
        out["lagged_innovation"] = out["innovation"].shift(1)
        out["innovation_mean"] = out["innovation"].rolling(40, min_periods=40).mean()
        out["innovation_std"] = out["innovation"].rolling(40, min_periods=40).std(ddof=0)
        out["innovation_autocorr"] = out["innovation"].rolling(40, min_periods=40).corr(
            out["lagged_innovation"]
        )
        scale = out["innovation_std"].where(out["innovation_std"] > 1e-12)
        out["innovation_z"] = (out["innovation"] - out["innovation_mean"]) / scale
        out["previous_z"] = out["innovation_z"].shift(1)
        out["previous_close"] = close.shift(1)
        ranges = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1)
        out["true_range"] = ranges.max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[["slope", "innovation_autocorr", "innovation_z", "previous_z", "atr"]].notna().all(axis=1)
        out["entry"] = (
            ready
            & (out["atr"] > 0)
            & (out["slope"] > 0)
            & (out["innovation_autocorr"] > 0.15)
            & (out["innovation_z"] > params.innovation_threshold)
            & (out["previous_z"] <= params.innovation_threshold)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        stop = np.nan
        age = 0

        for i in range(len(data)):
            if pending:
                pending = False
                # The previous close's decision fills at this bar's open.
                if np.isfinite(opens[i]) and opens[i] > 0:
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                    active = True
                    age = 0

            if active:
                age += 1
                # Freeze the entry ATR and entry price; never ratchet this stop.
                if not np.isfinite(closes[i]) or closes[i] < stop or age >= 8:
                    active = False
                else:
                    signal[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                signal[i] = 1
                pending = True

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Check delayed fills, fixed rather than trailing stops, and eight-session expiry.
    index = pd.date_range("2020-01-01", periods=18, freq="B")
    close = np.full(18, 102.0)
    close[1:5] = [103.0, 110.0, 101.0, 99.0]
    data = pd.DataFrame({
        "open": 102.0,
        "high": np.maximum(102.0, close) + 1.0,
        "low": np.minimum(102.0, close) - 1.0,
        "close": close,
        "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({"entry": False, "atr": 1.0}, index=index)
    ind.iloc[[0, 6], ind.columns.get_loc("entry")] = True
    result = GeneratedStrategy.generate_signals(data, ind, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert result["size"].gt(0).all()
    calculated = GeneratedStrategy.indicators(data, Params())
    assert calculated.index.equals(data.index)
    assert not calculated["entry"].any()
