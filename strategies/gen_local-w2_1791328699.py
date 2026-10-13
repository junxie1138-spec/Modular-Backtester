from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 12
    atr_window: int = 14
    entry_stretch: float = 1.25
    min_entropy: float = 0.80
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791328699"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(32 + params.energy_window - 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        scales = (2, 4, 8, 16)
        for half in scales:
            out[f"recent_{half}"] = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"older_{half}"] = out[f"recent_{half}"].shift(half)
            out[f"detail_{half}"] = (out[f"recent_{half}"] - out[f"older_{half}"]) / np.sqrt(2.0)
            out[f"center_{half}"] = (out[f"recent_{half}"] + out[f"older_{half}"]) / 2.0
            out[f"energy_{half}"] = out[f"detail_{half}"].pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
        out["total_energy"] = out[[f"energy_{h}" for h in scales]].sum(axis=1, min_count=4)
        out["positive_energy"] = out["total_energy"].where(out["total_energy"] > 0)
        for half in scales:
            out[f"weight_{half}"] = out[f"energy_{half}"] / out["positive_energy"]
            out[f"entropy_{half}"] = -out[f"weight_{half}"] * np.log(out[f"weight_{half}"].clip(lower=1e-12))
            out[f"anchor_{half}"] = out[f"weight_{half}"] * out[f"center_{half}"]
        out["entropy"] = out[[f"entropy_{h}" for h in scales]].sum(axis=1, min_count=4) / np.log(4.0)
        out["anchor"] = out[[f"anchor_{h}" for h in scales]].sum(axis=1, min_count=4)
        out["energy_radius"] = np.sqrt(out["positive_energy"])
        out["relative_position"] = (out["log_close"] - out["anchor"]) / out["energy_radius"]
        out["previous_close"] = data["close"].shift(1)
        out["intraday_range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["intraday_range", "high_gap", "low_gap"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        stretch = indicators["relative_position"].to_numpy(dtype=float)
        entropy = indicators["entropy"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25)
        direction = 0
        entry_price = np.nan
        entry_bar = -1
        entry_atr = 0.0
        conviction = 0.25
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if direction:
                # The previous close's order fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    entry_bar = i
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                stopped = valid and direction * (closes[i] - entry_price) < -params.stop_atr * entry_atr
                expired = i - entry_bar >= params.hold_bars - 1
                if not valid or stopped or expired:
                    direction = 0
                    entry_price = np.nan
            elif i >= warmup and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(stretch[i]) and np.isfinite(entropy[i]):
                if entropy[i] >= params.min_entropy and abs(stretch[i]) >= params.entry_stretch:
                    direction = -1 if stretch[i] > 0 else 1
                    entry_atr = atr[i]
                    entry_price = np.nan
                    conviction = float(0.25 + 0.75 * np.clip(abs(stretch[i]) / params.entry_stretch - 1.0, 0.0, 1.0))
            signals[i] = direction
            sizes[i] = conviction if direction else 0.25
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=80, freq="B")
    base = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0}, index=index)
    start = strategy.warmup_bars(params)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=start)
    for direction in (-1, 1):
        data = base.copy()
        ind = pd.DataFrame({"atr": 2.0, "relative_position": 0.0, "entropy": 1.0}, index=index)
        ind.iloc[start, ind.columns.get_loc("relative_position")] = -direction * 2.0
        ind.iloc[start + 1:, ind.columns.get_loc("atr")] = 100.0
        data.iloc[start + 1, data.columns.get_loc("open")] = 100.0 + direction * 5.0
        data.iloc[start + 2, data.columns.get_loc("close")] = 100.0 - direction
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].iloc[start] == 0
        assert result["signal"].iloc[start + 1:start + 3].eq(direction).all()
        assert result["signal"].iloc[start + 3] == 0
        assert np.isclose(result["size"].iloc[start + 1], 0.7)
        result = strategy.generate_signals(base, ind, ctx, params).data
        assert result["signal"].iloc[start + 1:start + 1 + params.hold_bars].eq(direction).all()
        assert result["signal"].iloc[start + 1 + params.hold_bars] == 0
    data = base.copy()
    data["close"] += 2.0 * np.sin(np.arange(len(data)) / 3.0)
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(full.iloc[:60], strategy.indicators(data.iloc[:60], params))
    assert full["relative_position"].iloc[start:].notna().all()


if __name__ == "__main__":
    _self_check()
