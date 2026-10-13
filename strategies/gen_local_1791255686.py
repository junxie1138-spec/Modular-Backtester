from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    signature_bars: int = 5
    compression_bars: int = 3
    baseline_bars: int = 20
    atr_bars: int = 14
    minimum_area: float = 0.08
    compression_ratio: float = 0.75
    breakeven_pct: float = 0.004
    trail_atr: float = 0.9


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791255686"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(2, params.signature_bars, params.compression_bars,
                   params.baseline_bars, params.atr_bars) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        n = max(2, params.signature_bars)
        short = max(1, params.compression_bars)
        long = max(1, params.baseline_bars)
        atr_n = max(1, params.atr_bars)
        log_price = np.log(data["close"].where(data["close"] > 0))
        displacement = log_price.diff()
        variation = displacement.abs().rolling(n, min_periods=n).sum()
        # Antisymmetric second-order time-price signature over n increments.
        area = (0.5 * (n + 1) * (log_price + log_price.shift(n))
                - log_price.rolling(n + 1, min_periods=n + 1).sum())
        signature = area.div(n * variation.replace(0.0, np.nan))
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        recent_range = true_range.rolling(short, min_periods=short).mean()
        baseline_range = true_range.rolling(long, min_periods=long).mean()
        compression = recent_range.div(baseline_range.replace(0.0, np.nan))
        atr = true_range.rolling(atr_n, min_periods=atr_n).mean()
        return pd.DataFrame({
            "log_price": log_price,
            "displacement": displacement,
            "variation": variation,
            "signature_area": area,
            "signature": signature,
            "true_range": true_range,
            "recent_range": recent_range,
            "baseline_range": baseline_range,
            "compression": compression,
            "atr": atr,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        recovery = ((indicators["signature"] > params.minimum_area)
                    & (indicators["displacement"] > 0)).fillna(False)
        compressed = (indicators["compression"] < params.compression_ratio).fillna(False)
        eligible = (recovery & compressed & indicators["atr"].gt(0)).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = stop = peak = 0.0
        sessions = 0
        k = max(0.01, float(params.trail_atr))
        trigger = max(0.0, float(params.breakeven_pct))
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry = opens[i]
                    peak = entry
                    stop = entry - k * atr[i - 1]
                    sessions = 0
                    armed = False
            if active:
                sessions += 1
                valid_bar = np.isfinite([opens[i], highs[i], lows[i], closes[i]]).all()
                # A stop observed during this session requests next-open exit.
                exit_now = not valid_bar or opens[i] <= stop or lows[i] <= stop
                if valid_bar and not exit_now:
                    peak = max(peak, highs[i])
                    if peak >= entry * (1.0 + trigger):
                        armed = True
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, peak - k * atr[i])
                        # Newly raised stops apply at the close and thereafter;
                        # the earlier low cannot establish intrabar ordering.
                        exit_now = closes[i] <= stop
                if exit_now or sessions >= 2:
                    active = False
                else:
                    decisions[i] = 1
            elif eligible[i]:
                pending = True
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0] * 5,
        "high": [100.2] * 5,
        "low": [99.95] * 5,
        "close": [100.1] * 5,
        "volume": [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        "signature": [0.2, 0.0, 0.0, 0.0, 0.0],
        "displacement": [0.001] * 5,
        "compression": [0.5] * 5,
        "atr": [1.0] * 5,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "high"] = 101.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    indicators["compression"] = 1.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
