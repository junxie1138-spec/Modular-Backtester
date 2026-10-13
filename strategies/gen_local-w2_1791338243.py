from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    ma_window: int = 20
    stretch_threshold: float = 1.5
    overlap_threshold: float = 0.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791338243"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    @staticmethod
    def warmup_bars(params: RecoveryParams) -> int:
        if params.ma_window < 3 or not 3 <= params.hold_bars <= 5:
            raise ValueError("ma_window must be >= 3; hold_bars must be 3 through 5")
        if not np.isfinite(params.stretch_threshold) or params.stretch_threshold <= 0:
            raise ValueError("stretch_threshold must be positive and finite")
        if not np.isfinite(params.overlap_threshold) or not 0 <= params.overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be between zero and one")
        return int(params.ma_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["intersection"] = (
            pd.concat([data["high"], data["high"].shift(1)], axis=1).min(axis=1, skipna=False)
            - pd.concat([data["low"], data["low"].shift(1)], axis=1).max(axis=1, skipna=False)
        ).clip(lower=0)
        out["overlap"] = (
            out["intersection"]
            / pd.concat([out["range"], out["range"].shift(1)], axis=1).min(axis=1, skipna=False).replace(0, np.nan)
        ).clip(0, 1)
        out["prior_z"] = out["z"].shift(1)
        out["prior_overlap"] = out["overlap"].shift(1)
        ready = out[["z", "prior_z", "prior_overlap"]].notna().all(axis=1)
        release = ready & (out["prior_overlap"] >= params.overlap_threshold) & (data["volume"] > 0) & (data["volume"] < data["volume"].shift(1))
        long_entry = (
            release & (out["prior_z"] <= -params.stretch_threshold)
            & (out["z"] > out["prior_z"]) & (out["z"] < 0)
            & (close > data["high"].shift(1))
        )
        short_entry = (
            release & (out["prior_z"] >= params.stretch_threshold)
            & (out["z"] < out["prior_z"]) & (out["z"] > 0)
            & (close < data["low"].shift(1))
        )
        out["entry_side"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        out["entry_size"] = (
            0.35 * out["prior_z"].abs() / params.stretch_threshold
        ).clip(lower=0.35, upper=0.95).fillna(0.35)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        entries = indicators["entry_side"].fillna(0).to_numpy(dtype=int)
        strengths = indicators["entry_size"].fillna(0.35).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.35, dtype=float)
        side, exit_bar, entry_size = 0, -1, 0.35
        for i in range(len(data)):
            if side != 0 and i == exit_bar:
                side = 0
                continue
            if side == 0 and entries[i] != 0:
                side = int(entries[i])
                exit_bar = i + params.hold_bars
                entry_size = float(np.clip(strengths[i], 0.35, 0.95))
            signals[i] = side
            if side != 0:
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.35).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry_side": [1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 0, 0], "entry_size": [0.45] * 5 + [0.8] * 7}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, 0]
    assert np.allclose(result["size"].iloc[1:5], 0.45)
    assert np.allclose(result["size"].iloc[6:10], 0.8)
    assert (result["size"] > 0).all()
    flat = GeneratedStrategy.indicators(data, params)
    assert (flat["entry_side"] == 0).all()
    assert GeneratedStrategy.warmup_bars(params) >= params.ma_window + 1


if __name__ == "__main__":
    _self_check()
