from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    history_bars: int = 5
    area_threshold: float = 0.15
    atr_bars: int = 14
    breakeven_pct: float = 0.006
    trail_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791231500"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(200, params.atr_bars + 2, params.history_bars + 4))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.history_bars < 2 or params.atr_bars < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not all(np.isfinite(x) and x > 0 for x in (params.area_threshold, params.breakeven_pct, params.trail_atr)):
            raise ValueError("Thresholds and stop parameters must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["range"] = data["high"] - data["low"]
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["scale"] = out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["price_step"] = data["close"].diff() / out["scale"]
        out["range_step"] = out["range"].diff() / out["scale"]
        # Antisymmetric second-order signature for two consecutive increments.
        out["area"] = 0.5 * (
            out["price_step"].shift(1) * out["range_step"]
            - out["range_step"].shift(1) * out["price_step"]
        )
        out["prior_area"] = out["area"].shift(1).rolling(params.history_bars, min_periods=params.history_bars).mean()
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[["atr", "area", "prior_area", "price_step", "range_step", "ma200"]].notna().all(axis=1)
        contracting = out["range_step"] < 0
        long_entry = (
            ready & contracting & (data["close"] > out["ma200"])
            & (out["price_step"] > 0)
            & (out["area"] > params.area_threshold)
            & (out["prior_area"] < -0.5 * params.area_threshold)
        )
        short_entry = (
            ready & contracting & (data["close"] < out["ma200"])
            & (out["price_step"] < 0)
            & (out["area"] < -params.area_threshold)
            & (out["prior_area"] > 0.5 * params.area_threshold)
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        requests = indicators["entry"].fillna(0).to_numpy(dtype=int)
        position = 0
        entry = np.nan
        stop = -np.inf
        peak = -np.inf
        armed = False
        age = 0
        for i in range(n):
            if not all(np.isfinite(x) and x > 0 for x in (opens[i], highs[i], lows[i], closes[i], atr[i])):
                position = 0
                entry = np.nan
                armed = False
                age = 0
                continue
            if position:
                # The preceding close's entry decision fills at this bar's open.
                if np.isnan(entry):
                    entry = opens[i]
                    stop = position * entry - params.trail_atr * atr[i - 1]
                    peak = position * entry
                age += 1
                # Signed coordinates make both long and short stops ratchet upward.
                adverse = lows[i] if position == 1 else -highs[i]
                favorable = highs[i] if position == 1 else -lows[i]
                hit = adverse <= stop
                peak = max(peak, favorable)
                armed = armed or peak - position * entry >= entry * params.breakeven_pct
                if armed:
                    stop = max(stop, position * entry, peak - params.trail_atr * atr[i])
                    # Newly raised stops are checked against the close, avoiding
                    # assumptions about the intraday ordering of high and low.
                    hit = hit or position * closes[i] <= stop
                if hit or age >= 2:
                    position = 0
                    entry = np.nan
                    age = 0
                    armed = False
                    continue
            else:
                position = int(requests[i])
                entry = np.nan
                age = 0
                armed = False
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": np.full(n, 0.95, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({
        "open": [100.0] * 8,
        "high": [100, 102, 101, 100, 100, 100.2, 99.5, 100],
        "low": [100, 99.8, 100, 100, 100, 98, 98, 100],
        "close": [100, 101.5, 100.5, 100, 100, 98.5, 99, 100],
        "volume": [1000.0] * 8,
    }, index=index)
    indicators = pd.DataFrame({"atr": [1.0] * 8, "entry": [1, 0, 0, 0, -1, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, SignatureParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    for end in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:end], indicators.iloc[:end], None, SignatureParams()).data
        assert prefix.equals(result.iloc[:end])
