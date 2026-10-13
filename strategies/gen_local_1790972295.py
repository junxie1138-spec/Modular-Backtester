from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationDebtParams:
    roc_bars: int = 3
    pressure_bars: int = 12
    min_drop: float = 0.01
    debt_ratio: float = 1.5
    atr_bars: int = 14
    initial_stop_atr: float = 2.5
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[AccelerationDebtParams]):
    strategy_id = "gen_local_1790972295"

    @classmethod
    def params_type(cls) -> type[AccelerationDebtParams]:
        return AccelerationDebtParams

    def warmup_bars(self, params: AccelerationDebtParams) -> int:
        return int(max(params.roc_bars + params.pressure_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: AccelerationDebtParams) -> pd.DataFrame:
        if min(params.roc_bars, params.pressure_bars, params.atr_bars, params.max_hold_bars) < 1:
            raise ValueError("Lookbacks and holding limit must be positive.")
        if not all(np.isfinite(v) and v > 0 for v in (
            params.min_drop, params.debt_ratio, params.initial_stop_atr,
            params.breakeven_pct, params.trail_atr,
        )):
            raise ValueError("Thresholds and stop distances must be finite and positive.")
        close = data["close"]
        roc = close.div(close.shift(params.roc_bars).replace(0, np.nan)).sub(1)
        acceleration = roc.diff()
        downside = (-acceleration.clip(upper=0)).rolling(params.pressure_bars).sum()
        upside = acceleration.clip(lower=0).rolling(params.pressure_bars).sum()
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_bars).mean()
        out = pd.DataFrame({
            "roc": roc, "acceleration": acceleration,
            "downside_pressure": downside, "upside_pressure": upside, "atr": atr,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)
        valid = out.notna().all(axis=1) & atr.gt(0)
        out["candidate"] = (
            valid & roc.le(-params.min_drop) & acceleration.gt(0)
            & downside.gt(params.debt_ratio * upside)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AccelerationDebtParams,
    ) -> SignalFrame:
        candidate = indicators["candidate"].fillna(False).astype(bool)
        confirmed = (candidate & candidate.shift(1, fill_value=False)).to_numpy()
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        armed = False
        entry = peak = stop = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if not (np.isfinite(opens[i]) and opens[i] > 0
                        and i > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                # The previous close's decision fills at this bar's open.
                entry = peak = opens[i]
                stop = entry - params.initial_stop_atr * atr[i - 1]
                entry_bar = i
                active = True
                armed = False
            if active:
                if not (np.isfinite(closes[i]) and closes[i] > 0
                        and np.isfinite(atr[i]) and atr[i] > 0):
                    active = False
                    continue
                # Stops are observed at closes; exits fill at the next open.
                if closes[i] <= stop or i - entry_bar + 1 >= params.max_hold_bars:
                    active = False
                    continue
                peak = max(peak, closes[i])
                armed = armed or closes[i] >= entry * (1 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                if closes[i] <= stop:
                    active = False
                    continue
                signal[i] = 1
            elif confirmed[i] and i + 1 < len(data):
                signal[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = AccelerationDebtParams()
    index = pd.date_range("2020-01-01", periods=32)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000000.0,
    }, index=index)
    derived = strategy.indicators(data, params)
    flat = strategy.generate_signals(data, derived, None, params).data
    assert flat["signal"].eq(0).all()
    assert strategy.warmup_bars(params) >= params.roc_bars + params.pressure_bars + 2
    sample = data.iloc[:9].copy()
    sample["close"] = [100, 100, 100, 95, 90, 102, 101, 99, 99]
    manual = pd.DataFrame({
        "candidate": [False, True, False, True, True, False, False, False, False],
        "atr": [1, 1, 1, 1, 1, 1, 10, 10, 10],
    }, index=sample.index)
    result = strategy.generate_signals(sample, manual, None, params).data
    # Isolated confirmation cannot enter; ATR expansion cannot loosen breakeven.
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
