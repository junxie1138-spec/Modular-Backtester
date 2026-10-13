from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RepairParams:
    lookback: int = 20
    volume_multiple: float = 1.3
    move_atr: float = 0.25
    gap_atr: float = 0.1
    target_annual_vol: float = 0.12
    breakeven_pct: float = 0.006
    trail_atr: float = 0.8
    max_hold: int = 2

    def __post_init__(self):
        if self.lookback < 2 or self.max_hold not in (1, 2):
            raise ValueError("lookback must be >= 2; max_hold must be 1 or 2")
        values = (self.volume_multiple, self.move_atr, self.gap_atr,
                  self.target_annual_vol, self.breakeven_pct, self.trail_atr)
        if not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError("thresholds and risk parameters must be finite and positive")


class GeneratedStrategy(BaseStrategy[RepairParams]):
    strategy_id = "gen_local_1791230908"

    @classmethod
    def params_type(cls):
        return RepairParams

    def warmup_bars(self, params: RepairParams) -> int:
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: RepairParams) -> pd.DataFrame:
        n = params.lookback
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = true_range.rolling(n, min_periods=n).mean()
        out["prior_atr"] = out["atr"].shift(1)
        volume_baseline = data["volume"].rolling(n, min_periods=n).mean().shift(1)
        out["volume_ratio"] = data["volume"] / volume_baseline.replace(0, np.nan)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["annual_vol"] = out["return"].rolling(n, min_periods=n).std(ddof=0) * np.sqrt(252.0)
        out["body"] = data["close"] - data["open"]
        out["previous_body"] = out["body"].shift(1)
        out["gap"] = data["open"] - previous
        out["previous_midpoint"] = ((data["open"] + data["close"]) / 2.0).shift(1)
        out["move_direction"] = np.sign(out["body"]).where(
            (out["volume_ratio"] >= params.volume_multiple)
            & (out["body"].abs() >= params.move_atr * out["prior_atr"]), 0
        )
        gap_direction = np.sign(out["gap"])
        out["repair_direction"] = gap_direction.where(
            (out["gap"] * out["previous_body"] < 0)
            & (out["gap"].abs() >= params.gap_atr * out["prior_atr"])
            & ((data["close"] - out["previous_midpoint"]) * gap_direction > 0), 0
        )
        out["candidate"] = out["move_direction"].where(
            out["move_direction"] == out["repair_direction"], 0
        )
        out["target_size"] = (
            params.target_annual_vol / out["annual_vol"].replace(0, np.nan)
        ).clip(lower=0.05, upper=0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RepairParams
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.05, dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        target_sizes = indicators["target_size"].to_numpy(dtype=float)
        direction = 0
        entry = 0.0
        stop = 0.0
        peak = 0.0
        held_size = 0.05
        age = 0
        armed = False
        warmup = self.warmup_bars(params)

        for i in range(count):
            # Yesterday's unshifted target executes at today's open.
            pending = signals[i - 1] if i else 0
            if direction and pending == 0:
                direction = 0
            valid_prices = np.isfinite(prices[i]).all() and (prices[i] > 0).all()
            if direction == 0 and pending != 0:
                if not valid_prices or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                    continue
                direction = int(pending)
                entry = prices[i, 0]
                held_size = sizes[i - 1]
                stop = -params.trail_atr * atr[i - 1] / entry
                peak = 0.0
                age = 0
                armed = False

            if direction:
                age += 1
                sizes[i] = held_size
                if not valid_prices or not np.isfinite(atr[i]) or atr[i] <= 0:
                    continue
                favorable_price = prices[i, 1] if direction == 1 else prices[i, 2]
                adverse_price = prices[i, 2] if direction == 1 else prices[i, 1]
                favorable = direction * (favorable_price / entry - 1.0)
                adverse = direction * (adverse_price / entry - 1.0)
                # Stop breaches schedule a flat target for the next open.
                if adverse <= stop or age >= params.max_hold:
                    continue
                peak = max(peak, favorable)
                armed = armed or peak >= params.breakeven_pct
                if armed:
                    # Signed-profit stop only rises; short price stops fall.
                    # Raised stops take effect next bar, avoiding intrabar ordering assumptions.
                    stop = max(stop, 0.0, peak - params.trail_atr * atr[i] / entry)
                signals[i] = direction
            elif i >= warmup - 1 and valid_prices:
                if candidates[i] != 0 and np.isfinite(target_sizes[i]):
                    signals[i] = candidates[i]
                    sizes[i] = float(target_sizes[i])

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=50, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5,
         "close": 100.0, "volume": 100.0}, index=index
    )
    data.iloc[19] = [100.0, 100.5, 98.5, 99.0, 100.0]
    data.iloc[20] = [100.0, 101.2, 99.8, 101.0, 1000.0]
    data.iloc[21] = [101.0, 101.8, 100.9, 101.2, 100.0]
    data.iloc[22] = [101.2, 101.3, 100.7, 100.8, 100.0]
    strategy = GeneratedStrategy()
    params = RepairParams()
    for direction in (1, -1):
        sample = data.copy()
        if direction == -1:
            sample["open"] = 200.0 - data["open"]
            sample["close"] = 200.0 - data["close"]
            sample["high"] = 200.0 - data["low"]
            sample["low"] = 200.0 - data["high"]
        derived = strategy.indicators(sample, params)
        result = strategy.generate_signals(sample, derived, None, params).data
        assert result["signal"].iloc[:21].eq(0).all()
        assert result["signal"].iloc[21:24].tolist() == [direction, direction, 0]
        assert result["size"].gt(0).all()
        # A one-session expiry must flatten on the following open.
        short_hold = RepairParams(max_hold=1)
        timed = strategy.generate_signals(
            sample, strategy.indicators(sample, short_hold), None, short_hold
        ).data
        assert timed["signal"].iloc[21:23].tolist() == [direction, 0]
        breached = sample.copy()
        breached.iloc[21, breached.columns.get_loc("low" if direction == 1 else "high")] = 99.0 if direction == 1 else 101.0
        stopped = strategy.generate_signals(
            breached, strategy.indicators(breached, params), None, params
        ).data
        assert stopped["signal"].iloc[21:23].tolist() == [direction, 0]


if __name__ == "__main__":
    _self_check()
