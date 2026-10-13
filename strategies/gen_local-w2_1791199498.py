from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    profit_target: float = 0.045
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791199498"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return 21

    def indicators(self, data, params):
        price = data["close"].where(
            np.isfinite(data["close"]) & (data["close"] > 0)
        )
        peak = pd.Series(0.0, index=data.index)
        trough = peak.copy()
        upper = pd.Series(-np.inf, index=data.index)
        lower = pd.Series(np.inf, index=data.index)
        # Endpoint horizontal visibility: intervening prices must clear both endpoints.
        for lag in range(1, 21):
            past = price.shift(lag)
            peak += (np.minimum(price, past) > upper).astype(float)
            trough += (np.maximum(price, past) < lower).astype(float)
            upper = np.maximum(upper, past)
            lower = np.minimum(lower, past)
        ready = price.rolling(21, min_periods=21).count().eq(21)
        quarter_open = pd.Series(
            data.index.month.isin([1, 4, 7, 10]) & (data.index.day <= 10),
            index=data.index,
        )
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[ready & quarter_open & (trough >= 5) & (trough - peak >= 3)] = 1
        candidate.loc[ready & quarter_open & (peak >= 5) & (peak - trough >= 3)] = -1
        return pd.DataFrame({
            "peak_degree": peak.where(ready),
            "trough_degree": trough.where(ready),
            "quarter_open": quarter_open,
            "candidate": candidate,
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        if not 0 < params.profit_target < 1:
            raise ValueError("profit_target must be between zero and one")
        if not 15 <= params.hold_bars <= 20:
            raise ValueError("hold_bars must be between 15 and 20")
        candidate = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                # The previous close's entry instruction fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                gain = (
                    side * (closes[i] / entry_price - 1.0)
                    if np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i]) else np.nan
                )
                if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                    side = 0
            elif candidate[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(candidate[i])
                entry_bar = i + 1
                entry_price = np.nan
            signal[i] = side
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2020-12-01", periods=60)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = Params(profit_target=0.04, hold_bars=15)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=21)
    pd.testing.assert_frame_equal(
        strategy.indicators(data.iloc[:40], params),
        strategy.indicators(data, params).iloc[:40],
    )
    assert strategy.indicators(data, params)["candidate"].eq(0).all()
    for side in (1, -1):
        derived = pd.DataFrame({"candidate": 0}, index=index)
        derived.iloc[21, 0] = side
        result = strategy.generate_signals(data, derived, ctx, params).data
        expected = np.zeros(60, dtype=int)
        expected[22:37] = side
        assert np.array_equal(result["signal"].to_numpy(), expected)
        changed = data.copy()
        changed.iloc[23, changed.columns.get_loc("close")] = 100.0 * (1 + side * 0.05)
        result = strategy.generate_signals(changed, derived, ctx, params).data
        expected[24:] = 0
        assert np.array_equal(result["signal"].to_numpy(), expected)
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
