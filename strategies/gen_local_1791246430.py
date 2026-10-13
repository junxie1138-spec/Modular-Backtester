from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockParams:
    corr_window: int = 80
    volume_window: int = 20
    atr_window: int = 14
    volume_multiple: float = 1.2
    corr_threshold: float = 0.2
    min_pairs: int = 8
    move_atr: float = 0.25
    activation_pct: float = 0.004
    trail_atr: float = 1.5
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ShockParams]):
    strategy_id = "gen_local_1791246430"

    @classmethod
    def params_type(cls):
        return ShockParams

    @staticmethod
    def warmup_bars(params: ShockParams) -> int:
        return int(max(200, params.volume_window + params.corr_window + 2,
                       params.atr_window + params.corr_window + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockParams) -> pd.DataFrame:
        if (min(params.corr_window, params.volume_window, params.atr_window) < 2
                or not 2 <= params.min_pairs <= params.corr_window
                or params.max_hold_bars not in (1, 2)
                or not 0 < params.corr_threshold <= 1
                or min(params.volume_multiple, params.activation_pct, params.trail_atr) <= 0
                or params.move_atr < 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ma200"] = close.rolling(200).mean()
        out["volume_baseline"] = data["volume"].rolling(params.volume_window).mean().shift(1)
        out["volume_ratio"] = data["volume"] / out["volume_baseline"].where(out["volume_baseline"] > 0)
        out["shock"] = ((out["volume_ratio"] >= params.volume_multiple)
                        & (out["return"].abs() >= params.move_atr * out["atr"] / previous)
                        & np.isfinite(out["return"]) & np.isfinite(out["atr"]))
        x, y = out["return"].shift(1), out["return"]
        eligible = out["shock"].shift(1, fill_value=False) & np.isfinite(x) & np.isfinite(y)
        out["pairs"] = eligible.astype(float).rolling(params.corr_window).sum()
        count = out["pairs"].where(out["pairs"] >= params.min_pairs)
        moments = pd.DataFrame({
            "x": x.where(eligible, 0.0),
            "y": y.where(eligible, 0.0),
            "xx": (x * x).where(eligible, 0.0),
            "yy": (y * y).where(eligible, 0.0),
            "xy": (x * y).where(eligible, 0.0)
        }, index=data.index).rolling(params.corr_window).sum().div(count, axis=0)
        out["mean_x"], out["mean_y"] = moments["x"], moments["y"]
        out["var_x"] = (moments["xx"] - moments["x"] ** 2).clip(lower=0)
        out["var_y"] = (moments["yy"] - moments["y"] ** 2).clip(lower=0)
        out["covariance"] = moments["xy"] - moments["x"] * moments["y"]
        denominator = np.sqrt(out["var_x"] * out["var_y"])
        out["correlation"] = (out["covariance"] / denominator.where(denominator > 0)).clip(-1, 1)
        out["forecast"] = out["mean_y"] + out["covariance"] / out["var_x"].where(out["var_x"] > 0) * (out["return"] - out["mean_x"])
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ShockParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        side = pending = age = 0
        entry = stop = peak = 0.0
        armed = False
        for i in range(len(data)):
            bar = data.iloc[i]
            ind = indicators.iloc[i]
            if side == 0 and pending:
                prior_atr = float(indicators["atr"].iloc[i - 1])
                if np.isfinite(bar["open"]) and bar["open"] > 0 and np.isfinite(prior_atr) and prior_atr > 0:
                    side = pending
                    entry = side * float(bar["open"])
                    peak = entry
                    stop = entry - params.trail_atr * prior_atr
                    age, armed = 0, False
                pending = 0
            valid = np.isfinite([bar["high"], bar["low"], bar["close"], ind["atr"], ind["ma200"]]).all() and ind["atr"] > 0
            if side:
                age += 1
                adverse = float(bar["low"]) if side == 1 else -float(bar["high"])
                favorable = float(bar["high"]) if side == 1 else -float(bar["low"])
                if (not valid or adverse <= stop or age >= params.max_hold_bars
                        or side * (bar["close"] - ind["ma200"]) <= 0):
                    side = 0
                else:
                    # Signed prices ratchet upward for both longs and shorts.
                    peak = max(peak, favorable)
                    armed = armed or peak >= entry + abs(entry) * params.activation_pct
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * float(ind["atr"]))
                    # The old stop tests the range; a new stop tests the close.
                    # This avoids assuming the order of today's high and low.
                    if side * float(bar["close"]) <= stop:
                        side = 0
                raw[i] = side
                continue
            if (valid and i + 1 >= GeneratedStrategy.warmup_bars(params)
                    and bool(ind["shock"]) and np.isfinite(ind["forecast"])
                    and np.isfinite(ind["correlation"])
                    and abs(ind["correlation"]) >= params.corr_threshold):
                candidate = int(np.sign(ind["forecast"]))
                if candidate * (bar["close"] - ind["ma200"]) > 0:
                    pending = candidate
                    raw[i] = candidate
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = ShockParams()
    index = pd.date_range("2020-01-01", periods=205)
    data = pd.DataFrame({"open": 100.0, "high": 100.1, "low": 99.9,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "ma200": 90.0, "shock": False,
                        "forecast": 0.01, "correlation": 0.5}, index=index)
    ind.loc[index[199], "shock"] = True
    def signals():
        return GeneratedStrategy.generate_signals(data, ind, None, params).data["signal"]
    assert signals().iloc[199:203].tolist() == [0, 1, 1, 0]
    data.loc[index[200], ["high", "close"]] = [101.0, 100.8]
    data.loc[index[201], ["open", "low", "close"]] = [99.8, 99.5, 99.8]
    assert signals().iloc[199:203].tolist() == [0, 1, 1, 0]
    # A retracement through the newly armed breakeven stop exits next open.
    data.loc[index[200], "close"] = 99.9
    assert signals().iloc[199:202].tolist() == [0, 1, 0]
    data.loc[index[200], ["high", "close"]] = [103.0, 101.0]
    assert signals().iloc[199:202].tolist() == [0, 1, 0]
    data.loc[index[200], ["high", "low", "close"]] = [100.1, 98.0, 99.0]
    assert signals().iloc[199:202].tolist() == [0, 1, 0]
    data.loc[:, ["open", "high", "low", "close"]] = [100.0, 100.1, 99.9, 100.0]
    ind["ma200"], ind["forecast"] = 110.0, -0.01
    assert signals().iloc[199:203].tolist() == [0, -1, -1, 0]
    data.loc[index[200], ["low", "close"]] = [99.0, 100.1]
    assert signals().iloc[199:202].tolist() == [0, -1, 0]
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not np.isinf(derived.select_dtypes(include="number").to_numpy()).any()
    prefix = GeneratedStrategy.indicators(data.iloc[:202], params)
    pd.testing.assert_frame_equal(prefix, derived.iloc[:202])


if __name__ == "__main__":
    _self_check()
