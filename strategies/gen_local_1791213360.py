from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    memory: int = 24
    stop_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791213360"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params):
        return int(max(15, params.memory + 5))

    def indicators(self, data, params):
        if params.memory < 8 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("memory must be >= 8 and stop_atr must be positive.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        for lag in (1, 2, 3):
            out[f"lag{lag}"] = out["return"].shift(lag)
        # Antisymmetric second-level signature of three (return, lagged return) increments.
        x0, y0 = out["lag2"], out["lag3"]
        x1, y1 = out["lag1"], out["lag2"]
        x2, y2 = out["return"], out["lag1"]
        out["signature_area"] = 0.5 * (
            x0 * y1 - y0 * x1 + x0 * y2 - y0 * x2 + x1 * y2 - y1 * x2
        )
        out["path_energy"] = x0**2 + y0**2 + x1**2 + y1**2 + x2**2 + y2**2
        out["area"] = out["signature_area"] / out["path_energy"].replace(0.0, np.nan)
        out["previous_area"] = out["area"].shift(1)
        out["serial_product"] = out["return"] * out["lag1"]
        out["lag_energy"] = out["lag1"]**2
        valid = out["previous_area"].notna() & out["serial_product"].notna()
        for label, mask in (
            ("positive", out["previous_area"] > 0),
            ("negative", out["previous_area"] < 0),
        ):
            out[f"{label}_count"] = mask.astype(float).where(valid).rolling(params.memory).sum()
            out[f"{label}_product"] = out["serial_product"].where(mask, 0.0).where(valid).rolling(params.memory).sum()
            out[f"{label}_energy"] = out["lag_energy"].where(mask, 0.0).where(valid).rolling(params.memory).sum()
            out[f"{label}_beta"] = out[f"{label}_product"] / out[f"{label}_energy"].replace(0.0, np.nan)
        positive = out["area"] > 0
        out["beta"] = out["positive_beta"].where(positive, out["negative_beta"])
        out["count"] = out["positive_count"].where(positive, out["negative_count"])
        out["forecast"] = out["beta"] * out["return"]
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14).mean()
        eligible = (
            (out["area"].abs() >= 0.10) & (out["beta"].abs() >= 0.15)
            & (out["count"] >= 4) & np.isfinite(out["forecast"])
        )
        out["intent"] = np.where(eligible, np.sign(out["forecast"].fillna(0.0)), 0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        intent = indicators["intent"].fillna(0).to_numpy(dtype=int)
        target = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry, fixed_atr = np.nan, np.nan
        for i in range(len(data)):
            if side:
                # Yesterday's entry decision fills at today's open.
                if age == 0:
                    entry = opens[i]
                age += 1
                invalid = not (np.isfinite(entry) and entry > 0 and np.isfinite(close[i]))
                stopped = side * (close[i] - entry) < -params.stop_atr * fixed_atr
                if invalid or stopped or age >= 2:
                    side, age = 0, 0
            elif (
                intent[i] != 0 and np.isfinite(close[i]) and close[i] > 0
                and np.isfinite(atr[i]) and atr[i] > 0
            ):
                side, age = int(intent[i]), 0
                fixed_atr = atr[i]
            target[i] = side
        df = pd.DataFrame({"signal": target, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0,
         "close": [100.0, 98.0, 100.0, 100.0, 102.0, 100.0, 100.0, 100.0],
         "volume": 1000.0}, index=index
    )
    strategy = GeneratedStrategy()
    params = SignatureParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=29)
    indicators = pd.DataFrame(
        {"intent": [1, 0, 0, -1, 0, 0, 1, 0],
         "atr": [1.0, 100.0, 100.0, 1.0, 100.0, 100.0, 1.0, 100.0]}, index=index
    )
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].tolist() == [0, 1, 0, 0, -1, 0, 0, 1]
    data["close"] = 100.0
    indicators["intent"] = 1
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0, 1]
    actual = strategy.indicators(data, params)
    assert actual.index.equals(index) and actual["intent"].eq(0).all()


if __name__ == "__main__":
    _self_check()
