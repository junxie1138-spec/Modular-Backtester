from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_window: int = 20
    z_entry: float = 0.5
    hold_bars: int = 2

    def __post_init__(self):
        if self.ma_window < 2 or self.hold_bars not in (1, 2):
            raise ValueError("ma_window must be >= 2; hold_bars must be 1 or 2")
        if not np.isfinite(self.z_entry) or self.z_entry <= 0:
            raise ValueError("z_entry must be finite and positive")


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1790975797"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(200, params.ma_window + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["anchor"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean().shift(1)
        out["dispersion"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0).shift(1)
        out["z_close"] = ((close - out["anchor"]) / out["dispersion"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["previous_high"] = data["high"].shift(1)
        out["island_clearance"] = data["low"] - out["previous_high"]
        out["entry"] = (
            (out["z_close"] <= -params.z_entry)
            & (out["island_clearance"] > 0.0)
            & (close > out["ma200"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, entry in enumerate(entries):
            if i < exit_bar:
                raw[i] = 1
            elif i == exit_bar:
                # Force a flat bar; an entry here cannot cancel the timed exit.
                raw[i] = 0
            elif entry:
                raw[i] = 1
                exit_bar = i + params.hold_bars

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    close = np.linspace(100.0, 160.0, 220)
    close[-6:] = [152.0, 154.0, 154.0, 154.0, 154.0, 154.0]
    data = pd.DataFrame(
        {"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=len(close), freq="B"),
    )
    strategy = GeneratedStrategy()
    for hold in (1, 2):
        params = GeneratedParams(hold_bars=hold)
        ind = strategy.indicators(data, params)
        assert ind["entry"].iloc[-5]
        assert not ind["entry"].iloc[:199].any()
        # Repeated entries cannot postpone the timed exit.
        ind["entry"] = False
        ind.iloc[-5:, ind.columns.get_loc("entry")] = True
        ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
        signal = strategy.generate_signals(data, ind, ctx, params).data["signal"]
        assert signal.iloc[-6:].tolist() == ([0, 0, 1, 0, 1, 0] if hold == 1 else [0, 0, 1, 1, 0, 1])


if __name__ == "__main__":
    _self_check()
