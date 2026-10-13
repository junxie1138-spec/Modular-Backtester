from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSlopeParams:
    slope_window: int = 12
    noise_window: int = 30
    atr_window: int = 14
    gap_atr: float = 0.25
    innovation_z: float = 0.75
    stop_atr: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapSlopeParams]):
    strategy_id = "gen_local_1791253754"

    @classmethod
    def params_type(cls):
        return GapSlopeParams

    def warmup_bars(self, params: GapSlopeParams) -> int:
        return int(max(200, params.slope_window + params.noise_window + 1,
                       params.atr_window + 2))

    def indicators(self, data: pd.DataFrame, params: GapSlopeParams) -> pd.DataFrame:
        if (params.slope_window < 2 or params.noise_window < 2
                or params.atr_window < 1 or params.holding_bars < 1
                or not all(np.isfinite(x) and x > 0 for x in
                           (params.gap_atr, params.innovation_z, params.stop_atr))):
            raise ValueError("Lookbacks and thresholds must be positive; slope and noise windows must be at least two.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["gap"] = data["open"] - previous
        ratio = close.where(close > 0) / data["open"].where(data["open"] > 0)
        out["intraday_log_return"] = np.log(ratio).replace([np.inf, -np.inf], np.nan)
        # Remove overnight jumps before estimating the latent intraday trend.
        out["gap_neutral_path"] = out["intraday_log_return"].fillna(0.0).cumsum()
        n = params.slope_window
        weights = np.arange(n, dtype=float) - (n - 1) / 2.0
        weights /= np.dot(weights, weights)
        path = out["gap_neutral_path"].to_numpy()
        slope = np.convolve(path, weights[::-1], mode="full")[:len(path)] if len(path) else path
        out["valid_slope_window"] = out["intraday_log_return"].notna().astype(int).rolling(
            n, min_periods=n
        ).sum().eq(n)
        out["latent_slope"] = pd.Series(slope, index=data.index).where(out["valid_slope_window"])
        out["slope_innovation"] = out["latent_slope"].diff()
        out["innovation_noise"] = out["slope_innovation"].shift(1).rolling(
            params.noise_window, min_periods=params.noise_window
        ).std(ddof=0)
        out["entry"] = (
            (close > out["ma200"])
            & (out["previous_atr"] > 0)
            & (out["gap"] < -params.gap_atr * out["previous_atr"])
            & (close >= previous)
            & (out["latent_slope"].shift(1) <= 0)
            & (out["latent_slope"] > 0)
            & (out["innovation_noise"] > 0)
            & (out["slope_innovation"] > params.innovation_z * out["innovation_noise"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapSlopeParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_atr = 0.0
        stop = 0.0
        age = 0
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            exited = False
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    # Freeze the stop using the entry-bar open and decision-bar ATR.
                    stop = opens[i] - params.stop_atr * entry_atr
                    active = True
                    age = 0
                else:
                    exited = True
            if active:
                age += 1
                if (not np.isfinite(closes[i]) or not np.isfinite(regime[i])
                        or closes[i] < stop or closes[i] <= regime[i]
                        or age >= params.holding_bars):
                    active = False
                    exited = True
            if (not active and not exited and entries[i]
                    and np.isfinite(atr[i]) and atr[i] > 0
                    and np.isfinite(closes[i]) and closes[i] > 0):
                entry_atr = atr[i]
                pending = True
            signals[i] = int(active or pending)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Checks next-open entry anchoring, frozen ATR, close-stop timing and one-bar shift.
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100, 100, 110, 110, 110, 100, 100],
        "close": [100, 100, 112, 109, 107, 100, 100],
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [2, 2, 2, 0.1, 0.1, 0.1, 0.1],
        "ma200": 90.0,
        "entry": [False, True, False, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(
        data, indicators, StrategyContext("SPY", "1d", 200), GapSlopeParams()
    )
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0]
