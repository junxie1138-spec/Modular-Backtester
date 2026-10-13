from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPersistenceParams:
    lookback: int = 60
    persistence_min: float = 0.20
    gap_z_min: float = 0.80
    profit_target: float = 0.02
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[GapPersistenceParams]):
    strategy_id = "gen_local_1791334273"

    @classmethod
    def params_type(cls):
        return GapPersistenceParams

    def warmup_bars(self, params):
        return max(200, params.lookback + 3)

    def indicators(self, data, params):
        if params.lookback < 3 or params.max_hold < 1:
            raise ValueError("lookback must be >= 3 and max_hold >= 1")
        if not (0 <= params.persistence_min < 1):
            raise ValueError("persistence_min must be in [0, 1)")
        if not np.isfinite(params.gap_z_min) or params.gap_z_min <= 0:
            raise ValueError("gap_z_min must be positive and finite")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1).where(lambda s: s > 0)
        opening = data["open"].where(lambda s: s > 0)
        out["gap"] = opening / prior_close - 1.0
        out["intraday"] = data["close"] / opening - 1.0
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        w = params.lookback
        gap = out["gap"]
        out["gap_mean"] = gap.rolling(w, min_periods=w).mean().shift(1)
        out["gap_std"] = gap.rolling(w, min_periods=w).std().shift(1)
        out["gap_autocorr"] = gap.rolling(w, min_periods=w).corr(gap.shift(1)).shift(1)
        out["counterflow_corr"] = gap.rolling(w, min_periods=w).corr(out["intraday"]).shift(1)
        out["gap_z"] = (gap - out["gap_mean"]) / out["gap_std"].where(out["gap_std"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (data["close"] > out["ma200"])
            & (out["gap_autocorr"] >= params.persistence_min)
            & (out["counterflow_corr"] < -0.20)
            & (out["gap_z"] >= params.gap_z_min)
            & (out["gap"] > 0)
            & (out["intraday"] < -0.5 * out["gap"])
            & (data["close"] > prior_close)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        eligible = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if entry_bar < 0:
                    # The prior close's entry decision fills at this open.
                    entry_bar = i
                    entry_price = opening[i]
                invalid = not np.isfinite(entry_price) or entry_price <= 0
                target_hit = np.isfinite(closing[i]) and closing[i] >= entry_price * (1 + params.profit_target)
                expired = i - entry_bar + 1 >= params.max_hold
                if invalid or target_hit or expired:
                    active = False
                    entry_bar = -1
                    entry_price = np.nan
                    # Exit takes precedence; do not re-enter on this close.
                    continue
                signal[i] = 1
            elif eligible[i]:
                active = True
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = GapPersistenceParams(max_hold=3)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    data.loc[index[2], "close"] = 103.0
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    warmup = strategy.indicators(data, params)
    assert not warmup["entry"].any()
    assert signals["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
