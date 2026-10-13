from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    shock_threshold: float = 0.15
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791254912"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 26

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.shock_threshold) or params.shock_threshold <= 0:
            raise ValueError("shock_threshold must be positive and finite")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        close = data["close"].astype(float)
        clock = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        price_sum = close.rolling(20, min_periods=20).sum()
        weighted_sum = (clock * close).rolling(20, min_periods=20).sum()
        slope = (weighted_sum - (clock - 9.5) * price_sum) / 665.0
        level = price_sum / 20.0 + 9.5 * slope
        innovation = slope.diff()
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean().replace(0.0, np.nan)
        impulse = innovation.rolling(5, min_periods=5).sum() / atr
        ready = (
            (impulse.shift(1) < -params.shock_threshold)
            & (impulse > impulse.shift(1))
            & (innovation > 0.0)
            & (close < level)
            & (slope.shift(5) > 0.0)
        ).fillna(False)
        return pd.DataFrame({
            "price_sum": price_sum,
            "weighted_sum": weighted_sum,
            "slope": slope,
            "level": level,
            "innovation": innovation,
            "true_range": true_range,
            "atr": atr,
            "impulse": impulse,
            "entry_ready": ready,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        stop = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    # The prior close requested entry; this bar supplies its fill open.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                    entry_bar = i
                    active = True
            exited = False
            if active:
                if not np.isfinite(closes[i]) or closes[i] < stop or i - entry_bar >= 17:
                    active = False
                    exited = True
                else:
                    signal[i] = 1
            if not active and not exited and ready[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending = True
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=25, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 2.0, "entry_ready": False}, index=index)
    indicators.loc[index[0], "entry_ready"] = True
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19:].eq(0).all()
    # A higher entry open must move the frozen stop to that actual entry price.
    data.loc[index[1], "open"] = 103.0
    data.loc[index[3], "close"] = 97.0
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].iloc[3] == 1
    assert result["signal"].iloc[4:].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
