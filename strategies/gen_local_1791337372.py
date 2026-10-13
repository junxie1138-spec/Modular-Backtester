from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 15
    target_vol: float = 0.12

    def __post_init__(self):
        if self.signature_window < 2:
            raise ValueError("signature_window must be at least 2")
        if not np.isfinite(self.target_vol) or self.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive")


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791337372"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params):
        return int(max(params.signature_window, 20) + 1)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        r = np.log(data["close"].where(data["close"] > 0)).diff()
        r = r.replace([np.inf, -np.inf], np.nan)
        downside = r.clip(upper=0).pow(2)
        x = pd.Series(0.0, index=data.index)
        y = pd.Series(0.0, index=data.index)
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature of (downside energy, return).
        for lag in range(params.signature_window - 1, -1, -1):
            dx = downside.shift(lag)
            dy = r.shift(lag)
            area = area + 0.5 * (x * dy - y * dx)
            x = x + dx
            y = y + dy
        length = r.abs().rolling(params.signature_window).sum()
        energy = r.pow(2).rolling(params.signature_window).sum()
        normalized = area / (x * length).replace(0.0, np.nan)
        snr = y / np.sqrt(energy).replace(0.0, np.nan)
        vol = r.rolling(20).std(ddof=1) * np.sqrt(252.0)
        out["log_return"] = r
        out["downside_energy"] = downside
        out["window_downside_energy"] = x
        out["window_return"] = y
        out["signature_area"] = area
        out["absolute_path_length"] = length
        out["return_energy"] = energy
        out["normalized_area"] = normalized
        out["signal_to_noise"] = snr
        out["annual_vol"] = vol
        out["entry_size"] = (params.target_vol / vol.where(vol > 0)).clip(
            lower=0.05, upper=0.95
        ).replace([np.inf, -np.inf], np.nan).fillna(0.05)
        out["entry"] = (
            normalized.gt(0.05) & snr.gt(0.6) & vol.gt(0)
            & np.isfinite(vol) & np.isfinite(normalized) & np.isfinite(snr)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        sizes = indicators["entry_size"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.05, dtype=float)
        exit_bar = -1
        held_size = 0.05
        first = self.warmup_bars(params) - 1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signal[i] = 1
                size[i] = held_size
            elif i >= first and entries[i]:
                held_size = float(sizes[i])
                exit_bar = i + 18
                signal[i] = 1
                size[i] = held_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SignatureParams(signature_window=3)
    index = pd.date_range("2020-01-01", periods=64)
    close = 100.0 * np.exp(np.resize([-0.02, 0.025, 0.025, 0.0], 64).cumsum())
    data = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": 1000.0}, index=index
    )
    derived = strategy.indicators(data, params)
    assert derived["signature_area"].iloc[6] > 0
    prefix = strategy.indicators(data.iloc[:40], params)
    pd.testing.assert_frame_equal(prefix, derived.iloc[:40])
    a = strategy.warmup_bars(params) - 1
    forced = pd.DataFrame({"entry": False, "entry_size": 0.5}, index=index)
    forced.iloc[a, forced.columns.get_loc("entry")] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=a + 1)
    result = strategy.generate_signals(data, forced, ctx, params).data
    assert result["signal"].iloc[:a + 1].eq(0).all()
    assert result["signal"].iloc[a + 1:a + 19].eq(1).all()
    assert result["signal"].iloc[a + 19:].eq(0).all()
    assert result["signal"].sum() == 18
    assert result["size"].iloc[a + 1:a + 19].eq(0.5).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
