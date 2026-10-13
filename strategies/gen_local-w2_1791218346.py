from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ImpulseParams:
    participation: float = 1.25
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[ImpulseParams]):
    strategy_id = "gen_local-w2_1791218346"

    @classmethod
    def params_type(cls) -> type[ImpulseParams]:
        return ImpulseParams

    @staticmethod
    def warmup_bars(params: ImpulseParams) -> int:
        return 26

    @staticmethod
    def indicators(data: pd.DataFrame, params: ImpulseParams) -> pd.DataFrame:
        if not (np.isfinite(params.participation) and params.participation > 0
                and np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("Parameters must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        volume = data["volume"]
        out["previous_close"] = close.shift(1)
        out["return"] = close / out["previous_close"].replace(0, np.nan) - 1.0
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["ma20"] = close.rolling(20, min_periods=20).mean()
        out["ma_slope"] = out["ma20"].diff()
        out["volume_baseline"] = volume.rolling(20, min_periods=20).mean().shift(6)
        out["impulse_volume"] = volume.rolling(3, min_periods=3).mean().shift(3)
        out["recoil_volume"] = volume.rolling(2, min_periods=2).mean().shift(1)
        out["anchor"] = close.shift(6)
        out["impulse_end"] = close.shift(3)
        out["impulse_gain"] = out["impulse_end"] / out["anchor"].replace(0, np.nan) - 1.0
        out["impulse_path"] = out["return"].abs().rolling(3, min_periods=3).sum().shift(3)
        out["impulse_efficiency"] = out["impulse_gain"] / out["impulse_path"].replace(0, np.nan)
        out["retained_gain"] = (
            (out["previous_close"] - out["anchor"])
            / (out["impulse_end"] - out["anchor"]).replace(0, np.nan)
        )
        out["previous_high"] = data["high"].shift(1)
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out["entry"] = (
            (out["volume_baseline"] > 0)
            & (out["impulse_gain"] > 0)
            & (out["impulse_efficiency"] >= 0.65)
            & (out["impulse_volume"] >= params.participation * out["volume_baseline"])
            & (out["recoil_volume"] <= 0.75 * out["impulse_volume"])
            & (out["retained_gain"] >= 0.70)
            & (out["retained_gain"] < 1.0)
            & (close > out["previous_high"])
            & (volume >= params.participation * out["volume_baseline"])
            & (close > out["ma20"])
            & (out["ma_slope"] > 0)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ImpulseParams,
    ) -> SignalFrame:
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        signal = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atrs = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        live = armed = False
        entry = peak = stop = 0.0
        age = 0
        for i in range(len(data)):
            atr = atrs[i]
            # A prior close's entry decision fills at this bar's open.
            if not live and i > 0 and signal[i - 1] == 1:
                entry = float(opens[i])
                peak = entry
                stop = entry - 2.0 * float(atrs[i - 1])
                age = 0
                live, armed = True, False
            if live:
                age += 1
                # Evaluate the previously established stop before updating it.
                # Exit decisions execute next bar through the mandatory shift.
                if lows[i] <= stop or age >= 4:
                    live = False
                    continue
                peak = max(peak, float(highs[i]))
                armed = armed or peak >= entry * 1.01
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atr) and atr > 0:
                        stop = max(stop, peak - params.trail_atr * atr)
                signal[i] = 1
            elif entries[i] and np.isfinite(atr) and atr > 0:
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": 100.0,
        "high": [100.0, 100.4, 102.0, 101.0, 100.0, 100.0],
        "low": [99.9, 99.9, 100.2, 100.4, 99.9, 99.9],
        "close": 100.0,
        "volume": 100.0,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 1.0,
        "entry": [True, False, False, False, False, False],
    }, index=index)
    params = ImpulseParams()
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    data["high"] = 100.4
    data["low"] = 99.9
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    data["low"] = 97.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, 0]
    for n in range(1, len(data) + 1):
        prefix = GeneratedStrategy.generate_signals(
            data.iloc[:n], indicators.iloc[:n], None, params
        ).data
        assert prefix["signal"].tolist() == result["signal"].iloc[:n].tolist()
    warmup = GeneratedStrategy.indicators(data, params)
    assert not warmup["entry"].any()


if __name__ == "__main__":
    _self_check()
