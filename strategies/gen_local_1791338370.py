from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    ma_bars: int = 40
    flow_bars: int = 8
    entry_z: float = 1.6
    flow_threshold: float = 0.15
    atr_bars: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = "gen_local_1791338370"

    @classmethod
    def params_type(cls):
        return ElasticParams

    def warmup_bars(self, params: ElasticParams) -> int:
        return int(max(params.ma_bars, params.flow_bars, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: ElasticParams) -> pd.DataFrame:
        if min(params.ma_bars, params.flow_bars, params.atr_bars) < 2:
            raise ValueError("Indicator windows must be at least two bars.")
        if not (params.entry_z > 0 and 0 < params.flow_threshold < 1
                and params.breakeven_pct > 0 and params.trail_atr > 0
                and params.max_hold_bars >= 1):
            raise ValueError("Invalid entry, stop, or holding parameters.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out["std"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].replace(0, np.nan)
        out["range"] = data["high"] - data["low"]
        out["clv"] = ((2 * close - data["high"] - data["low"])
                      / out["range"].replace(0, np.nan)).where(out["range"].ne(0), 0.0)
        out["volume"] = data["volume"].clip(lower=0)
        out["signed_volume"] = out["clv"] * out["volume"]
        out["flow_sum"] = out["signed_volume"].rolling(
            params.flow_bars, min_periods=params.flow_bars).sum()
        out["volume_sum"] = out["volume"].rolling(
            params.flow_bars, min_periods=params.flow_bars).sum()
        out["flow"] = out["flow_sum"] / out["volume_sum"].replace(0, np.nan)
        out["previous_close"] = close.shift(1)
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["tr"] = out[["range", "high_gap", "low_gap"]].max(axis=1, skipna=False)
        out["atr"] = out["tr"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ElasticParams,
    ) -> SignalFrame:
        z = indicators["z"].to_numpy(dtype=float)
        flow = indicators["flow"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        entries = np.select(
            [(z <= -params.entry_z) & (flow >= params.flow_threshold),
             (z >= params.entry_z) & (flow <= -params.flow_threshold)],
            [1, -1], default=0,
        )
        decisions = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry, stop, best = np.nan, -np.inf, -np.inf
        armed = False
        for i in range(len(data)):
            valid = (np.isfinite(close[i]) and close[i] > 0
                     and np.isfinite(atr[i]) and atr[i] > 0)
            if side:
                # A decision from the previous close fills at this bar's open.
                if np.isnan(entry):
                    if not (np.isfinite(opens[i]) and opens[i] > 0):
                        side = 0
                        continue
                    entry = opens[i]
                    best = side * entry
                    stop = best - params.trail_atr * atr[i - 1]
                    age = 0
                    armed = False
                age += 1
                if not valid:
                    side = 0
                else:
                    price = side * close[i]
                    # Stops are close-triggered exits filled at the next open.
                    hit = price <= stop
                    best = max(best, price)
                    if side * (close[i] / entry - 1.0) >= params.breakeven_pct:
                        armed = True
                    if armed:
                        # Signed-price stop always rises; raw short stops fall.
                        stop = max(stop, side * entry, best - params.trail_atr * atr[i])
                    if hit or price <= stop or age >= params.max_hold_bars:
                        side = 0
                decisions[i] = side
                # Exit decisions cannot also reopen on the same close.
                continue
            if (valid and i + 1 >= self.warmup_bars(params)
                    and np.isfinite(z[i]) and np.isfinite(flow[i])):
                side = int(entries[i])
                if side:
                    entry = np.nan
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = ElasticParams(ma_bars=2, flow_bars=2, atr_bars=2)
    for side in (1, -1):
        prices = 100.0 + side * np.array([0, 0, 0, 0, 3, 0, 0], dtype=float)
        data = pd.DataFrame(
            {"open": 100.0, "high": prices + 1, "low": prices - 1,
             "close": prices, "volume": 1000.0},
            index=pd.date_range("2020-01-01", periods=7),
        )
        ind = pd.DataFrame({"z": 0.0, "flow": 0.0, "atr": 1.0}, index=data.index)
        ind.iloc[2, ind.columns.get_loc("z")] = -side * 2.0
        ind.iloc[2, ind.columns.get_loc("flow")] = side * 0.5
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, 0, 0, side, side, side, 0]
        prefix = strategy.generate_signals(data.iloc[:5], ind.iloc[:5], None, params).data
        assert prefix.equals(result.iloc[:5])
        flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
        flat_ind = strategy.indicators(flat, params)
        assert not np.isinf(flat_ind.to_numpy()).any()
        assert (strategy.generate_signals(flat, flat_ind, None, params).data["signal"] == 0).all()
        held = pd.DataFrame(
            {"open": 100.0, "close": 100.0},
            index=pd.date_range("2020-01-01", periods=24),
        )
        held_ind = pd.DataFrame({"z": 0.0, "flow": 0.0, "atr": 1.0}, index=held.index)
        held_ind.iloc[2, held_ind.columns.get_loc("z")] = -side * 2.0
        held_ind.iloc[2, held_ind.columns.get_loc("flow")] = side * 0.5
        timed = strategy.generate_signals(held, held_ind, None, params).data["signal"]
        assert (timed.iloc[3:21] == side).all() and (timed.iloc[21:] == 0).all()


if __name__ == "__main__":
    _self_check()
