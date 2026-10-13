from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 12
    atr_window: int = 14
    baseline_window: int = 60
    compression_limit: float = 0.40
    area_threshold: float = 0.035
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    hold_bars: int = 18
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790971953"

    @classmethod
    def params_type(cls):
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return int(max(params.lookback + 1, params.atr_window + params.baseline_window))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        n = params.lookback
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = np.log(data["open"].where(data["open"] > 0) / out["previous_close"].where(out["previous_close"] > 0))
        out["session"] = np.log(data["close"].where(data["close"] > 0) / data["open"].where(data["open"] > 0))
        out["cumulative_gap"] = out["gap"].fillna(0.0).cumsum()
        out["cumulative_session"] = out["session"].fillna(0.0).cumsum()
        out["area_step"] = out["cumulative_gap"].shift(1) * out["session"] - out["cumulative_session"].shift(1) * out["gap"]
        out["gap_sum"] = out["gap"].rolling(n, min_periods=n).sum()
        out["session_sum"] = out["session"].rolling(n, min_periods=n).sum()
        out["gap_variation"] = out["gap"].abs().rolling(n, min_periods=n).sum()
        out["session_variation"] = out["session"].abs().rolling(n, min_periods=n).sum()
        out["area"] = 0.5 * (out["area_step"].rolling(n, min_periods=n).sum() - out["cumulative_gap"].shift(n) * out["session_sum"] + out["cumulative_session"].shift(n) * out["gap_sum"])
        out["normalized_area"] = out["area"] / (0.5 * out["gap_variation"] * out["session_variation"]).replace(0.0, np.nan)
        out["session_imbalance"] = (out["session_sum"] - out["gap_sum"]) / (out["gap_variation"] + out["session_variation"]).replace(0.0, np.nan)
        out["strength"] = (out["normalized_area"].clip(-1.0, 1.0) * out["session_imbalance"].clip(-1.0, 1.0)).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat([data["high"] - data["low"], (data["high"] - out["previous_close"]).abs(), (data["low"] - out["previous_close"]).abs()], axis=1).max(axis=1)
        out["true_range"] = out["true_range"].where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["atr_baseline"] = out["atr"].rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out["range_span"] = data["high"].rolling(n, min_periods=n).max() - data["low"].rolling(n, min_periods=n).min()
        out["range_budget"] = out["true_range"].rolling(n, min_periods=n).sum()
        out["compression"] = out["range_span"] / out["range_budget"].replace(0.0, np.nan)
        out["ready"] = (np.arange(len(data)) >= GeneratedStrategy.warmup_bars(params) - 1) & out[["strength", "atr", "atr_baseline", "compression"]].notna().all(axis=1)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        side = 0
        entry = None
        stop = 0.0
        entry_atr = 0.0
        active_size = params.min_size
        age = 0
        armed = False
        for i in range(len(data)):
            row = data.iloc[i]
            ind = indicators.iloc[i]
            if side:
                # A decision from the preceding close fills at this bar's open.
                if entry is None:
                    entry = float(row["open"])
                    stop = side * entry - params.trail_atr * entry_atr
                    age = 0
                    armed = False
                age += 1
                adverse_price = float(row["low"] if side == 1 else -row["high"])
                # Test the previous stop before updating it from this close.
                if adverse_price <= stop or age >= params.hold_bars:
                    side = 0
                    entry = None
                    age = 0
                    armed = False
                else:
                    favorable_return = side * (float(row["close"]) / entry - 1.0)
                    if favorable_return >= params.breakeven_pct:
                        armed = True
                    if armed:
                        stop = max(stop, side * entry)
                        atr = float(ind["atr"])
                        if np.isfinite(atr) and atr > 0.0:
                            # The stop rises monotonically in direction-adjusted prices.
                            stop = max(stop, side * float(row["close"]) - params.trail_atr * atr)
            elif bool(ind["ready"]):
                strength = float(ind["strength"])
                atr = float(ind["atr"])
                compressed = float(ind["compression"]) <= params.compression_limit and atr <= float(ind["atr_baseline"])
                if compressed and atr > 0.0 and abs(strength) >= params.area_threshold:
                    side = 1 if strength > 0.0 else -1
                    entry = None
                    entry_atr = atr
                    conviction = min(1.0, abs(strength) / (3.0 * params.area_threshold))
                    active_size = float(params.min_size + (params.max_size - params.min_size) * conviction)
            signal[i] = side
            sizes[i] = active_size if side else params.min_size
        df["signal"] = signal
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=4)
    for direction in (1, -1):
        close = np.array([100.0, 100.0 + direction * 3.0, 100.0 - direction, 100.0])
        data = pd.DataFrame({"open": 100.0, "high": np.maximum(close, 100.0) + 0.5, "low": np.minimum(close, 100.0) - 0.5, "close": close, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"ready": [True, False, False, False], "strength": direction * 0.1, "atr": 1.0, "atr_baseline": 2.0, "compression": 0.2}, index=index)
        result = GeneratedStrategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0]
        assert result["size"].gt(0.0).all()
        assert result["size"].iloc[1] > params.min_size
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=100))
    ind = GeneratedStrategy.indicators(flat, params)
    assert GeneratedStrategy.generate_signals(flat, ind, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
