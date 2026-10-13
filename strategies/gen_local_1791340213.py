from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityResetParams:
    volatility_ratio: float = 0.65
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[VolatilityResetParams]):
    strategy_id = "gen_local_1791340213"

    @classmethod
    def params_type(cls):
        return VolatilityResetParams

    def warmup_bars(self, params: VolatilityResetParams) -> int:
        # One return lag, 20 baseline observations, five-bar shift, one crossing lag.
        return 27

    def indicators(self, data: pd.DataFrame, params: VolatilityResetParams) -> pd.DataFrame:
        if not 0.0 < params.volatility_ratio < 1.0:
            raise ValueError("volatility_ratio must lie strictly between zero and one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["fast_vol"] = out["return"].rolling(5, min_periods=5).std(ddof=0)
        out["baseline_vol"] = out["return"].rolling(20, min_periods=20).std(ddof=0).shift(5)
        out["vol_ratio"] = out["fast_vol"] / out["baseline_vol"].where(out["baseline_vol"] > 0.0)
        out["drift_snr"] = out["return"].rolling(5, min_periods=5).sum() / (
            out["fast_vol"].where(out["fast_vol"] > 0.0) * np.sqrt(5.0)
        )
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry"] = (
            out["vol_ratio"].lt(params.volatility_ratio)
            & out["vol_ratio"].shift(1).ge(params.volatility_ratio)
            & out["drift_snr"].ge(0.5)
            & out["atr"].gt(0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolatilityResetParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        pending = active = armed = False
        entry = stop = peak = entry_atr = 0.0
        entry_bar = -1

        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    active = True
                    armed = False
                    entry = peak = opens[i]
                    entry_bar = i
                    stop = max(0.0, entry - 2.0 * entry_atr)

            if active:
                valid = (
                    np.isfinite(closes[i]) and closes[i] > 0.0
                    and np.isfinite(lows[i]) and lows[i] > 0.0
                    and np.isfinite(atrs[i]) and atrs[i] > 0.0
                )
                # A breach of the previously established stop exits at the next open.
                if not valid or lows[i] <= stop or i - entry_bar >= 9:
                    active = False
                    continue
                peak = max(peak, closes[i])
                if closes[i] >= entry * 1.01:
                    armed = True
                if armed:
                    # New stop levels apply only after this close; no intrabar ordering assumptions.
                    stop = max(stop, entry, peak - params.trail_atr * atrs[i])
                decisions[i] = 1
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0.0:
                pending = True
                entry_atr = atrs[i]
                decisions[i] = 1

        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VolatilityResetParams()
    index = pd.date_range("2020-01-01", periods=14, freq="B")
    close = np.array([100.0, 100.5, 101.2, 103.0, 102.5, 100.5] + [100.0] * 8)
    opens = np.r_[100.0, close[:-1]]
    data = pd.DataFrame(
        {"open": opens, "high": np.maximum(opens, close) + 0.2,
         "low": np.minimum(opens, close) - 0.1,
         "close": close, "volume": 1000.0},
        index=index,
    )
    ind = pd.DataFrame({"atr": 1.0, "entry": False}, index=index)
    ind.loc[index[0], "entry"] = True
    ind.loc[index[4], "atr"] = 5.0
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=27)
    signals = strategy.generate_signals(data, ind, ctx, params).data
    # Rising ATR cannot lower the trail; the fifth held session breaches it.
    assert signals["signal"].tolist() == [0, 1, 1, 1, 1, 1] + [0] * 8
    assert signals["size"].gt(0.0).all()
    breakeven = data.copy()
    breakeven.loc[index[3], "low"] = 99.8
    assert strategy.generate_signals(breakeven, ind, ctx, params).data["signal"].iloc[4] == 0
    initial_stop = data.copy()
    initial_stop.loc[index[2], "low"] = 97.5
    assert strategy.generate_signals(initial_stop, ind, ctx, params).data["signal"].iloc[3] == 0
    flat = data.assign(open=100.0, high=100.1, low=99.9, close=100.0)
    expiry = strategy.generate_signals(flat, ind, ctx, params).data["signal"]
    assert expiry.tolist() == [0] + [1] * 10 + [0] * 3
    prefix = strategy.generate_signals(data.iloc[:6], ind.iloc[:6], ctx, params).data
    pd.testing.assert_frame_equal(prefix, signals.iloc[:6])
    constant = pd.concat([flat] * 3, ignore_index=True)
    constant.index = pd.date_range("2020-01-01", periods=len(constant), freq="B")
    derived = strategy.indicators(constant, params)
    assert not derived["entry"].any()
    assert strategy.generate_signals(constant, derived, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
