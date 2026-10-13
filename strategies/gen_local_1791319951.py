from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TideParams:
    streak_bars: int = 4
    target_vol: float = 0.12

    def __post_init__(self):
        if self.streak_bars < 1:
            raise ValueError("streak_bars must be positive")
        if not np.isfinite(self.target_vol) or self.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive")


class GeneratedStrategy(BaseStrategy[TideParams]):
    strategy_id = "gen_local_1791319951"

    @classmethod
    def params_type(cls):
        return TideParams

    @staticmethod
    def warmup_bars(params):
        return int(max(19, 18 + params.streak_bars - 1))

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        out["channel_high"] = data["high"].rolling(18, min_periods=18).max()
        out["channel_low"] = data["low"].rolling(18, min_periods=18).min()
        out["midpoint"] = (out["channel_high"] + out["channel_low"]) / 2.0
        out["phase"] = np.select(
            [data["close"] > out["midpoint"], data["close"] < out["midpoint"]],
            [1, -1], default=0,
        )
        groups = out["phase"].ne(out["phase"].shift()).cumsum()
        out["streak"] = (out.groupby(groups).cumcount() + 1).where(out["phase"].ne(0), 0)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["realized_vol"] = out["return"].rolling(18, min_periods=18).std(ddof=0) * np.sqrt(252.0)
        out["candidate"] = out["phase"].where(
            out["streak"].ge(params.streak_bars) & out["realized_vol"].notna(), 0,
        ).astype(int)
        out["exposure"] = (
            params.target_vol / out["realized_vol"].clip(lower=0.0001)
        ).clip(lower=0.05, upper=0.95).fillna(0.95)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx: StrategyContext, params):
        position = 0
        signals = np.zeros(len(data), dtype=int)
        for i, candidate in enumerate(indicators["candidate"].to_numpy(dtype=int)):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["exposure"].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = TideParams()
    index = pd.date_range("2020-01-01", periods=40)
    close = pd.Series(np.arange(100.0, 140.0), index=index)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.0})
    derived = GeneratedStrategy.indicators(data, params)
    assert derived["candidate"].iloc[:20].eq(0).all()
    assert derived["candidate"].iloc[20:].eq(1).all()
    pd.testing.assert_frame_equal(derived.iloc[:30], GeneratedStrategy.indicators(data.iloc[:30], params))
    cues = pd.DataFrame({"candidate": [1, 0, -1, -1, 0, 1, 1],
                         "exposure": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}, index=index[:7])
    result = GeneratedStrategy.generate_signals(data.iloc[:7], cues, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert np.allclose(result["size"], [0.95, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


if __name__ == "__main__":
    _self_check()
