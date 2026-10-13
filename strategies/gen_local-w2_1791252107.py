from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRegimeParams:
    atr_bars: int = 14
    regime_bars: int = 40
    correlation_cutoff: float = 0.20
    gap_threshold: float = 0.20
    stop_atr: float = 2.0
    hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[GapRegimeParams]):
    strategy_id = "gen_local-w2_1791252107"

    @classmethod
    def params_type(cls) -> type[GapRegimeParams]:
        return GapRegimeParams

    def warmup_bars(self, params: GapRegimeParams) -> int:
        return int(params.atr_bars + params.regime_bars + 2)

    def indicators(self, data: pd.DataFrame, params: GapRegimeParams) -> pd.DataFrame:
        if params.atr_bars < 1 or params.regime_bars < 3:
            raise ValueError("Lookbacks must be positive; regime_bars must be at least 3.")
        if not 0 < params.correlation_cutoff < 1:
            raise ValueError("correlation_cutoff must lie between zero and one.")
        if params.gap_threshold <= 0 or params.stop_atr <= 0 or params.hold_bars < 1:
            raise ValueError("Gap threshold, stop distance and holding period must be positive.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["prior_atr"] = out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["gap"] = (data["open"] - out["prior_close"]) / out["prior_atr"]
        out["session_move"] = (data["close"] - data["open"]) / out["prior_atr"]
        out["coupling"] = out["gap"].rolling(params.regime_bars).corr(
            out["session_move"]
        ).shift(1).clip(-1.0, 1.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["regime"] = np.select([
            out["coupling"] > params.correlation_cutoff,
            out["coupling"] < -params.correlation_cutoff,
        ], [1, -1], default=0).astype(int)
        direction = np.sign(out["gap"]).fillna(0).astype(int) * out["regime"]
        eligible = (
            (out["gap"].abs() >= params.gap_threshold)
            & (direction * out["session_move"] > 0)
            & out["atr"].gt(0)
        )
        out["candidate"] = direction.where(eligible, 0).astype(int)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].ne(0)
            & out["candidate"].eq(out["candidate"].shift(1))
            & out["regime"].eq(out["regime"].shift(1)),
            0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRegimeParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # The preceding close selected the trade; this open is its fill proxy.
                    stop = opens[i] - position * params.stop_atr * atr[i - 1]
                stopped = (
                    np.isfinite(closes[i]) and np.isfinite(stop)
                    and position * (closes[i] - stop) < 0
                )
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    position = 0
                    stop = np.nan
                signals[i] = position
                # An exit always produces a flat next bar before another entry.
                continue
            if entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapRegimeParams()
    index = pd.date_range("2020-01-01", periods=14)
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 96.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    for side in (1, -1):
        derived = pd.DataFrame({
            "atr": 1.0, "confirmed": [0, side] + [0] * 12,
        }, index=index)
        path = data.copy()
        path.loc[index[2], "close"] = 100.0 + side * 3.0
        path.loc[index[3], "close"] = 100.0 - side
        path.loc[index[4], "close"] = 100.0 - side * 3.0
        result = strategy.generate_signals(path, derived, None, params).data
        assert result["signal"].tolist() == [0, 0, side, side, side] + [0] * 9
        result = strategy.generate_signals(data, derived, None, params).data
        assert result["signal"].tolist() == [0, 0] + [side] * 10 + [0, 0]
        assert result["size"].gt(0).all()
    flat = strategy.indicators(data, params)
    assert flat["confirmed"].eq(0).all()


if __name__ == "__main__":
    _self_check()
