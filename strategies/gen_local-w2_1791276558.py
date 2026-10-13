from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    lookback: int = 40
    min_autocorr: float = 0.10
    min_gap: float = 0.0003
    profit_target: float = 0.008
    max_hold: int = 2
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = "gen_local-w2_1791276558"

    @classmethod
    def params_type(cls):
        return GapParams

    def warmup_bars(self, params: GapParams) -> int:
        # Prior close, lagged gap, and the second confirmation extend the window.
        return params.lookback + 3

    def indicators(self, data: pd.DataFrame, params: GapParams) -> pd.DataFrame:
        if not (
            params.lookback >= 3
            and 0.0 <= params.min_autocorr < 1.0
            and 0.0 <= params.min_gap < 1.0
            and 0.0 < params.profit_target < 1.0
            and params.max_hold in (1, 2)
            and 0.0 < params.size <= 1.0
        ):
            raise ValueError("Invalid gap strategy parameters")
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1).where(lambda s: s > 0.0)
        out["prior_close"] = prior_close
        out["gap"] = (data["open"] / prior_close - 1.0).replace(
            [np.inf, -np.inf], np.nan
        )
        out["lagged_gap"] = out["gap"].shift(1)
        out["gap_autocorr"] = out["gap"].rolling(
            params.lookback, min_periods=params.lookback
        ).corr(out["lagged_gap"]).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        out["qualified"] = (
            (out["gap"] > params.min_gap)
            & (out["gap_autocorr"] > params.min_autocorr)
        ).fillna(False)
        out["confirmed"] = out["qualified"] & out["qualified"].shift(
            1, fill_value=False
        )
        out["entry_ready"] = out["confirmed"] & (
            out["gap"] < out["lagged_gap"]
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                # Read the fill price only after its bar has actually arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                held = i - entry_bar + 1
                # Close-observed target; both exits execute at the following open.
                exit_now = (
                    not np.isfinite(entry_price)
                    or entry_price <= 0.0
                    or not np.isfinite(closes[i])
                    or closes[i] >= entry_price * (1.0 + params.profit_target)
                    or held >= params.max_hold
                )
                if exit_now:
                    active = False
                else:
                    raw[i] = 1
            elif ready[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                active = True
                entry_bar = i + 1
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapParams(lookback=3)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": [100., 100., 110., 110., 110., 110.],
         "high": [101., 101., 113., 113., 113., 113.],
         "low": [99., 99., 109., 109., 109., 109.],
         "close": [100., 100., 110., 110., 110., 110.],
         "volume": 1000.},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=6)
    evidence = pd.DataFrame(
        {"entry_ready": [False, True, False, False, False, False]}, index=index
    )
    assert strategy.generate_signals(data, evidence, ctx, params).data["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    data.loc[index[2], "close"] = 112.
    assert strategy.generate_signals(data, evidence, ctx, params).data["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.
    result = strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params)
    assert result.data["signal"].eq(0).all() and result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
