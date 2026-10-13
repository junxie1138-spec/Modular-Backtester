from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_window: int = 5
    volume_window: int = 20
    atr_window: int = 14
    area_threshold: float = 0.08
    stop_atr: float = 1.5
    hold_bars: int = 2
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791340148"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(params.volume_window + params.path_window,
                       params.atr_window + 1, params.path_window + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if (params.path_window < 2 or params.volume_window < 2
                or params.atr_window < 1 or params.hold_bars not in (1, 2)
                or not 0.0 < params.area_threshold < 1.0
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0
                or not 0.0 < params.min_size <= params.max_size <= 1.0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["return_step"] = np.log(close).diff()
        out["volume_mean"] = volume.rolling(params.volume_window).mean()
        out["participation_step"] = volume.div(out["volume_mean"].where(out["volume_mean"] > 0)) - 1.0
        dx = out["return_step"]
        dy = out["participation_step"]
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric level-two signature: half the sum of ordered cross-products.
        for lag in range(1, params.path_window):
            cross = dx.shift(lag) * dy - dy.shift(lag) * dx
            area = area + 0.5 * cross.rolling(params.path_window - lag).sum()
        out["signature_area"] = area
        out["return_variation"] = dx.abs().rolling(params.path_window).sum()
        out["participation_variation"] = dy.abs().rolling(params.path_window).sum()
        denominator = out["return_variation"] * out["participation_variation"]
        out["area_score"] = area.div(denominator.where(denominator > 0))
        out["momentum"] = dx.rolling(params.path_window).sum()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["strength"] = (out["area_score"].abs() / (3.0 * params.area_threshold)).clip(0.0, 1.0)
        out["target_size"] = params.min_size + (params.max_size - params.min_size) * out["strength"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        score = indicators["area_score"]
        valid = indicators[["area_score", "momentum", "atr", "target_size"]].notna().all(axis=1)
        valid &= indicators["atr"] > 0
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[valid & (score >= params.area_threshold) & (indicators["momentum"] > 0)] = 1
        candidate.loc[valid & (score <= -params.area_threshold) & (indicators["momentum"] < 0)] = -1
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        sizes = indicators["target_size"].to_numpy(dtype=float)
        candidates = candidate.to_numpy()
        signals = np.zeros(len(data), dtype=int)
        allocations = np.full(len(data), params.min_size, dtype=float)
        direction = 0
        entry_bar = -1
        stop = np.nan
        allocation = params.min_size
        for i in range(len(data)):
            if direction:
                if entry_bar == -1:
                    # Yesterday's decision fills at today's open; freeze its ATR.
                    if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0:
                        direction = 0
                        continue
                    entry_bar = i
                    stop = opens[i] - direction * params.stop_atr * atr[i - 1]
                stopped = (not np.isfinite(closes[i])
                           or (direction == 1 and closes[i] < stop)
                           or (direction == -1 and closes[i] > stop))
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    direction = 0
                    entry_bar = -1
                    allocation = params.min_size
            elif candidates[i] != 0:
                direction = int(candidates[i])
                entry_bar = -1
                allocation = float(sizes[i])
            signals[i] = direction
            allocations[i] = allocation
        df = pd.DataFrame({"signal": signals, "size": allocations}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 96.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"area_score": [0.2, 0.0, 0.0, 0.0, 0.0],
                            "momentum": 0.01, "atr": 1.0, "target_size": 0.7}, index=index)
    strategy = GeneratedStrategy()
    params = SignatureParams()
    held = strategy.generate_signals(data, derived, None, params).data
    assert held["signal"].tolist() == [0, 1, 1, 0, 0]
    assert held.loc[index[1], "size"] == 0.7
    data.loc[index[1], "close"] = 97.0
    stopped = strategy.generate_signals(data, derived, None, params).data
    assert stopped["signal"].tolist() == [0, 1, 0, 0, 0]
    derived["area_score"] = -derived["area_score"]
    derived["momentum"] = -derived["momentum"]
    data.loc[index[1], "close"] = 103.0
    short = strategy.generate_signals(data, derived, None, params).data
    assert short["signal"].tolist() == [0, -1, 0, 0, 0]
    derived["area_score"] = np.nan
    flat = strategy.generate_signals(data, derived, None, params).data
    assert flat["signal"].eq(0).all() and flat["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
