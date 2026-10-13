from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    graph_window: int = 12
    recovery_window: int = 30
    flow_window: int = 5
    atr_window: int = 14
    gap_jump: float = 0.04
    flow_threshold: float = 0.10
    min_drawdown: float = 0.025
    min_rebound: float = 0.006
    initial_stop_atr: float = 2.5
    breakeven_pct: float = 0.012
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790866640"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.recovery_window, params.graph_window + 2,
                       params.flow_window + 1, params.atr_window + 1))

    @staticmethod
    def _spectral_gap(windows: np.ndarray) -> np.ndarray:
        rows, width = windows.shape
        # ponytail: dense O(bars * window**2) storage; batch windows for larger histories.
        adjacency = np.zeros((rows, width, width), dtype=float)
        for left in range(width - 1):
            blocker = np.full(rows, -np.inf)
            for right in range(left + 1, width):
                slope = (windows[:, right] - windows[:, left]) / (right - left)
                visible = slope > blocker
                adjacency[:, left, right] = visible
                adjacency[:, right, left] = visible
                blocker = np.maximum(blocker, slope)
        scale = 1.0 / np.sqrt(adjacency.sum(axis=2))
        laplacian = np.eye(width)[None, :, :] - (
            adjacency * scale[:, :, None] * scale[:, None, :]
        )
        return np.clip(np.linalg.eigvalsh(laplacian)[:, 1], 0.0, 2.0)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if (params.graph_window < 3 or params.recovery_window < 2
                or params.flow_window < 2 or params.atr_window < 2
                or params.max_hold < 1
                or min(params.initial_stop_atr, params.breakeven_pct,
                       params.trail_atr) <= 0):
            raise ValueError("Invalid lookbacks or stop parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window).mean()
        out["peak"] = close.rolling(params.recovery_window).max()
        out["trough"] = close.rolling(params.recovery_window).min()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["rebound"] = close / out["trough"] - 1.0
        volume = data["volume"].astype(float).where(data["volume"] >= 0)
        out["signed_volume"] = np.sign(close.diff()) * volume
        out["flow"] = (
            out["signed_volume"].rolling(params.flow_window).sum()
            / volume.rolling(params.flow_window).sum().replace(0.0, np.nan)
        )
        out["gap"] = np.nan
        width = params.graph_window
        if len(data) >= width:
            windows = np.lib.stride_tricks.sliding_window_view(
                np.log(close.to_numpy()), width
            )
            valid = np.isfinite(windows).all(axis=1)
            gaps = np.full(len(windows), np.nan)
            if valid.any():
                gaps[valid] = self._spectral_gap(windows[valid])
            out.iloc[width - 1:, out.columns.get_loc("gap")] = gaps
        out["gap_change"] = out["gap"].diff(2)
        out["topology_ok"] = out["gap_change"] >= params.gap_jump
        out["flow_ok"] = out["flow"] >= params.flow_threshold
        out["entry"] = (
            out["topology_ok"] & out["flow_ok"]
            & (out["drawdown"] >= params.min_drawdown)
            & (out["rebound"] >= params.min_rebound)
            & (out["atr"] > 0)
            & np.isfinite(out[["atr", "gap", "gap_change", "flow"]]).all(axis=1)
        ).fillna(False)
        out.iloc[:self.warmup_bars(params) - 1, out.columns.get_loc("entry")] = False
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RecoveryParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        desired = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atrs = indicators["atr"].to_numpy(dtype=float)
        bars = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        holding = pending = armed = False
        entry = stop = peak = pending_atr = 0.0
        age = 0
        for i, (open_px, high, low, close) in enumerate(bars):
            if pending:
                pending = False
                if not np.isfinite(open_px) or open_px <= 0:
                    continue
                # Initialize only when the previously decided entry reaches its fill bar.
                entry = peak = open_px
                stop = entry - params.initial_stop_atr * pending_atr
                holding, armed, age = True, False, 0
            if holding:
                age += 1
                atr = atrs[i]
                if not np.isfinite([high, low, close, atr]).all() or atr <= 0:
                    holding = False
                    continue
                # Test the established stop before raising it with this bar's high.
                exited = low <= stop
                if not exited:
                    peak = max(peak, high)
                    armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr)
                    exited = close <= stop or age >= params.max_hold
                if exited:
                    holding = False
                else:
                    desired[i] = 1
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                pending, pending_atr = True, atrs[i]
                desired[i] = 1
        df["signal"] = desired
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    line = np.arange(12, dtype=float)
    gaps = strategy._spectral_gap(np.stack([line, line ** 2]))
    assert gaps[0] < 0.1 and gaps[1] > 1.0
    index = pd.date_range("2020-01-01", periods=6)
    sample = pd.DataFrame({
        "open": [100, 100, 102, 104, 102, 101],
        "high": [101, 102, 105, 104, 103, 102],
        "low": [99, 99.9, 102, 103.5, 101, 100],
        "close": [100, 101, 104, 104, 102, 101],
        "volume": [1000] * 6,
    }, index=index)
    forced = pd.DataFrame({
        "entry": [True, False, False, False, False, False],
        "atr": [1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
    }, index=index)
    context = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(sample, forced, context, RecoveryParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    assert result.data["size"].gt(0).all()
    capped = strategy.generate_signals(
        sample, forced, context, RecoveryParams(max_hold=2)
    )
    assert capped.data["signal"].tolist() == [0, 1, 1, 0, 0, 0]
