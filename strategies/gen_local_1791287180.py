from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SlopeShockParams:
    slope_window: int = 8
    shock_window: int = 5
    noise_window: int = 20
    shock_threshold: float = 0.8


class GeneratedStrategy(BaseStrategy[SlopeShockParams]):
    strategy_id = "gen_local_1791287180"

    @classmethod
    def params_type(cls) -> type[SlopeShockParams]:
        return SlopeShockParams

    def warmup_bars(self, params: SlopeShockParams) -> int:
        return int(max(200, params.slope_window + params.shock_window, params.noise_window + 2))

    def indicators(self, data: pd.DataFrame, params: SlopeShockParams) -> pd.DataFrame:
        if params.slope_window < 2 or params.shock_window < 1 or params.noise_window < 2:
            raise ValueError("Invalid lookback length")
        if not np.isfinite(params.shock_threshold) or params.shock_threshold <= 0:
            raise ValueError("shock_threshold must be finite and positive")
        close = data["close"].astype(float).where(lambda s: np.isfinite(s) & s.gt(0))
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["log_close"] = np.log(close)
        out["log_return"] = out["log_close"].diff()
        out["time"] = np.arange(len(data), dtype=float)
        n = params.slope_window
        out["slope"] = out["log_close"].rolling(n, min_periods=n).cov(out["time"], ddof=0) / ((n * n - 1) / 12.0)
        out["slope_innovation"] = out["slope"].diff()
        out["noise"] = out["log_return"].rolling(params.noise_window, min_periods=params.noise_window).std(ddof=0)
        out["prior_slope_noise"] = out["noise"].shift(1) / np.sqrt(float(n))
        out["accumulated_innovation"] = out["slope_innovation"].rolling(params.shock_window, min_periods=params.shock_window).sum()
        out["shock_score"] = out["accumulated_innovation"] / out["prior_slope_noise"].where(out["prior_slope_noise"].gt(1e-12))
        positive = out["shock_score"].gt(params.shock_threshold) & out["slope"].gt(0)
        negative = out["shock_score"].lt(-params.shock_threshold) & out["slope"].lt(0)
        out["event"] = np.select([positive, negative], [1, -1], default=0).astype(int)
        out["long_entry"] = positive & close.gt(out["ma200"])
        out["short_entry"] = negative & close.lt(out["ma200"])
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SlopeShockParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        state = 0
        events = indicators["event"].fillna(0).to_numpy(dtype=int)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        for i, (event, long_entry, short_entry) in enumerate(zip(events, longs, shorts)):
            # Opposing shocks exit even when MA200 blocks the opposite entry.
            if state != 0 and event == -state:
                state = 0
            elif state == 0:
                if long_entry:
                    state = 1
                elif short_entry:
                    state = -1
            signals[i] = state
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame({
        "event": [0, 1, 0, -1, -1, 0, 1, 1],
        "long_entry": [False, True, False, False, False, False, False, True],
        "short_entry": [False, False, False, False, True, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, SlopeShockParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
