from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    gap_multiple: float = 2.0
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = "gen_local-w2_1790965494"

    @classmethod
    def params_type(cls) -> type[GapParams]:
        return GapParams

    def warmup_bars(self, params: GapParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: GapParams) -> pd.DataFrame:
        prior_close = data["close"].shift(1).where(lambda s: s > 0)
        gap = (data["open"] / prior_close - 1).replace([np.inf, -np.inf], np.nan)
        noise = gap.shift(1).rolling(20, min_periods=20).std(ddof=0).clip(lower=0.001)
        dormant = gap.abs().shift(1).rolling(3, min_periods=3).max()
        prior_high = data["high"].shift(1)
        tr = pd.concat([data["high"] - data["low"], (data["high"] - prior_close).abs(), (data["low"] - prior_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        entry = ((gap >= params.gap_multiple * noise) & (dormant < 0.5 * noise) & (data["low"] > prior_high) & (data["close"] >= data["open"]) & (atr > 0)).fillna(False)
        return pd.DataFrame({"prior_close": prior_close, "gap": gap, "noise": noise, "dormant": dormant, "prior_high": prior_high, "true_range": tr, "atr": atr, "entry": entry}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapParams) -> SignalFrame:
        if not np.isfinite(params.gap_multiple) or not np.isfinite(params.trail_atr) or params.gap_multiple <= 0 or params.trail_atr <= 0:
            raise ValueError("Parameters must be finite and positive")
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        pending = active = armed = False
        entry_price = stop = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0:
                    # Match the configured five-basis-point entry slippage.
                    entry_price = opens[i] * 1.0005
                    stop = entry_price - 2.0 * atrs[i - 1]
                    entry_bar = i
                    active = True
                    armed = False
            if active:
                close = closes[i]
                # Stops are close-confirmed; orders execute at the next open.
                if i - entry_bar >= 17 or (np.isfinite(close) and close <= stop):
                    active = False
                    continue
                if np.isfinite(close):
                    armed = armed or close >= entry_price * 1.02
                    if armed:
                        stop = max(stop, entry_price)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, close - params.trail_atr * atrs[i])
                signals[i] = 1
            elif entries[i] and i >= self.warmup_bars(params) - 1:
                pending = True
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=44)
    close = np.full(44, 100.0)
    close[23:27] = [103.0, 104.0, 101.0, 100.0]
    data = pd.DataFrame({"open": 100.0, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "entry": False}, index=index)
    ind.loc[index[21], "entry"] = True
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[21:28].tolist() == [0, 1, 1, 1, 1, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[22:40].eq(1).all()
    assert result["signal"].iloc[40:].eq(0).all()
    data.loc[index[23], "close"] = 102.1
    data.loc[index[24], "close"] = 99.9
    ind["atr"] = 2.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[24] == 1 and result["signal"].iloc[25] == 0
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["entry"].iloc[:21].any()
    assert (result["size"] > 0).all() and set(result["signal"]) <= {0, 1}


if __name__ == "__main__":
    self_check()
