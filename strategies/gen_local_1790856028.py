from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 12
    regime_window: int = 60
    stress_ratio: float = 1.3
    area_threshold: float = 0.06
    pressure_threshold: float = 0.12
    profit_target: float = 0.04
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790856028"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(params.signature_window + 1, params.regime_window))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.signature_window < 2 or params.regime_window < params.signature_window:
            raise ValueError("Require regime_window >= signature_window >= 2")
        if params.stress_ratio <= 0 or params.area_threshold < 0:
            raise ValueError("Invalid regime or area threshold")
        if not 0 <= params.pressure_threshold < 1:
            raise ValueError("pressure_threshold must be in [0, 1)")
        w = params.signature_window
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        high = data["high"].where(data["high"] > 0)
        low = data["low"].where(data["low"] > 0)
        out["overnight"] = np.log(opening / closing.shift(1))
        out["intraday"] = np.log(closing / opening)
        x = out["overnight"]
        y = out["intraday"]
        out["signature_area"] = 0.0
        # Exact second-level antisymmetric signature over the last w increments.
        for lag in range(1, w):
            cross = x.shift(lag) * y - y.shift(lag) * x
            out["signature_area"] += 0.5 * cross.rolling(w - lag, min_periods=w - lag).sum()
        out["area_scale"] = x.abs().rolling(w, min_periods=w).sum() * y.abs().rolling(w, min_periods=w).sum()
        out["normalized_area"] = out["signature_area"] / out["area_scale"].where(out["area_scale"] > 0)
        out["net_return"] = (x + y).rolling(w, min_periods=w).sum()
        out["log_range"] = np.log(high / low).where(high >= low)
        out["fast_range"] = out["log_range"].rolling(w, min_periods=w).mean()
        out["slow_range"] = out["log_range"].rolling(params.regime_window, min_periods=params.regime_window).mean()
        out["range_ratio"] = out["fast_range"] / out["slow_range"].where(out["slow_range"] > 0)
        out["stressed"] = out["range_ratio"] > params.stress_ratio
        out["oriented_area"] = out["normalized_area"].where(~out["stressed"], -out["normalized_area"])
        spread = (high - low).where(high >= low)
        location = (2.0 * closing - high - low) / spread.where(spread > 0)
        out["close_location"] = location.where(spread != 0, 0.0).clip(-1.0, 1.0)
        out["valid_volume"] = data["volume"].where(data["volume"] >= 0)
        out["pressure_volume"] = out["close_location"] * out["valid_volume"]
        out["volume_sum"] = out["valid_volume"].rolling(w, min_periods=w).sum()
        out["pressure_sum"] = out["pressure_volume"].rolling(w, min_periods=w).sum()
        out["pressure"] = out["pressure_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[["oriented_area", "net_return", "range_ratio", "pressure"]].notna().all(axis=1)
        out["signature_agrees"] = (out["oriented_area"] > params.area_threshold) & (out["net_return"] > 0)
        out["pressure_agrees"] = out["pressure"] > params.pressure_threshold
        out["entry"] = ready & out["signature_agrees"] & out["pressure_agrees"]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        if params.hold_bars < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Require positive profit_target and hold_bars")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Reference the next session's open only once that bar arrives.
                    entry_price = opens[i]
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.hold_bars
                if invalid_entry or target_hit or time_up:
                    active = False
            elif entries[i]:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
            raw[i] = int(active)
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 99.0, "close": [100.0, 100.0, 105.0, 100.0, 100.0, 100.0], "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": [True, False, False, False, False, False]}, index=index)
    result = strategy.generate_signals(data, entries, None, SignatureParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, entries, None, SignatureParams(hold_bars=3)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
