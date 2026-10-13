from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeWaveParams:
    range_lookback: int = 10
    atr_lookback: int = 14
    expansion: float = 1.3
    stationary_limit: float = 0.15
    translation_min: float = 0.35
    close_edge: float = 0.25
    breakeven_pct: float = 0.004
    trail_atr: float = 0.9
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[RangeWaveParams]):
    strategy_id = "gen_local-w2_1791306626"

    @classmethod
    def params_type(cls) -> type[RangeWaveParams]:
        return RangeWaveParams

    def warmup_bars(self, params: RangeWaveParams) -> int:
        return int(max(200, params.range_lookback + 1, params.atr_lookback + 1))

    def indicators(self, data: pd.DataFrame, params: RangeWaveParams) -> pd.DataFrame:
        if not (
            params.range_lookback >= 1 and params.atr_lookback >= 1
            and params.expansion > 1
            and 0 <= params.stationary_limit < params.translation_min
            and 0 < params.close_edge < 0.5
            and params.breakeven_pct > 0 and params.trail_atr > 0
            and 1 <= params.max_hold <= 2
        ):
            raise ValueError("Invalid range-wave parameters")
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["range"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"]
        )
        out["range_baseline"] = out["range"].shift(1).rolling(
            params.range_lookback, min_periods=params.range_lookback
        ).mean()
        out["expansion"] = out["range"] / out["range_baseline"]
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["translation"] = out["midpoint"].diff() / out["range"]
        out["close_location"] = (data["close"] - data["low"]) / out["range"]
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_lookback, min_periods=params.atr_lookback
        ).mean()
        out["entry_atr"] = out["atr"].shift(1)
        valid = (
            out.notna().all(axis=1)
            & np.isfinite(out).all(axis=1)
            & (out["atr"] > 0) & (out["entry_atr"] > 0)
            & (data["close"] > 0)
        )
        expanded = out["expansion"] >= params.expansion
        centered = out["translation"].abs() <= params.stationary_limit
        lower = out["close_location"] <= params.close_edge
        upper = out["close_location"] >= 1.0 - params.close_edge
        long_setup = (data["close"] > out["ma200"]) & (
            (centered & lower)
            | ((out["translation"] >= params.translation_min) & upper)
        )
        short_setup = (data["close"] < out["ma200"]) & (
            (centered & upper)
            | ((out["translation"] <= -params.translation_min) & lower)
        )
        out["setup"] = np.select(
            [valid & expanded & long_setup, valid & expanded & short_setup],
            [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RangeWaveParams
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        stop = -np.inf
        peak = -np.inf
        armed = False
        age = 0
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        features = indicators[["setup", "atr", "entry_atr", "ma200"]].to_numpy(dtype=float)
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            setup, atr, entry_atr, ma200 = features[i]
            usable = (
                np.isfinite(prices[i]).all() and np.isfinite(features[i]).all()
                and min(opening, high, low, close) > 0
                and low <= min(opening, close) <= max(opening, close) <= high
                and atr > 0 and entry_atr > 0
            )
            if not usable:
                position = 0
                entry = np.nan
                continue
            if position:
                # A decision on i-1 is filled at this bar's open.
                if np.isnan(entry):
                    entry = opening
                    peak = position * entry
                    stop = peak - params.trail_atr * entry_atr
                    armed = False
                    age = 0
                age += 1
                # Signed prices make the favorable stop ratchet upward for both sides.
                favorable = high if position == 1 else -low
                adverse = low if position == 1 else -high
                signed_close = position * close
                touched = adverse <= stop
                peak = max(peak, favorable)
                if peak - position * entry >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, position * entry, peak - params.trail_atr * atr)
                regime_lost = (
                    (position == 1 and close <= ma200)
                    or (position == -1 and close >= ma200)
                )
                # New stops are checked against the close, not earlier intrabar lows/highs.
                # All observed exits become next-open orders through the final shift.
                if touched or (armed and signed_close <= stop) or regime_lost or age >= params.max_hold:
                    position = 0
                    entry = np.nan
                decisions[i] = position
                continue
            if setup:
                position = int(setup)
                entry = np.nan
                decisions[i] = position
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RangeWaveParams()
    assert strategy.warmup_bars(params) >= 200
    for direction in (1, -1):
        for early_exit in (False, True):
            data = pd.DataFrame(
                {"open": 95.0, "high": 105.0, "low": 95.0,
                 "close": 95.0, "volume": 1000.0},
                index=pd.date_range("2020-01-01", periods=204)
            )
            data.iloc[200, :4] = [96.0, 112.0, 92.0, 96.0]
            data.iloc[201, :4] = [97.0, 99.0, 96.0, 96.5 if early_exit else 98.0]
            data.iloc[202, :4] = [98.0, 100.0, 96.0, 97.5]
            if direction == -1:
                original = data.copy()
                data["open"] = 200.0 - original["open"]
                data["close"] = 200.0 - original["close"]
                data["high"] = 200.0 - original["low"]
                data["low"] = 200.0 - original["high"]
            features = strategy.indicators(data, params)
            result = strategy.generate_signals(data, features, None, params).data
            assert result.index.equals(data.index)
            assert result["signal"].iloc[:201].eq(0).all()
            assert result["signal"].iloc[201] == direction
            assert result["signal"].iloc[202] == (0 if early_exit else direction)
            assert result["signal"].iloc[203] == 0
            assert result["size"].gt(0).all()
            assert result["signal"].isin([-1, 0, 1]).all()


if __name__ == "__main__":
    _self_check()
