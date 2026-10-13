from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnAnchorParams:
    anchor_window: int = 60
    displacement_window: int = 5
    entry_band: float = 0.35
    atr_window: int = 14
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ReturnAnchorParams]):
    strategy_id = "gen_local_1790917773"

    @classmethod
    def params_type(cls) -> type[ReturnAnchorParams]:
        return ReturnAnchorParams

    def warmup_bars(self, params: ReturnAnchorParams) -> int:
        return int(max(params.anchor_window, params.displacement_window, params.atr_window) + 3)

    def indicators(self, data: pd.DataFrame, params: ReturnAnchorParams) -> pd.DataFrame:
        if (
            min(params.anchor_window, params.displacement_window, params.atr_window) < 2
            or not 0.0 < params.entry_band < 0.5
            or not np.isfinite(params.stop_atr)
            or params.stop_atr <= 0.0
            or not 3 <= params.hold_bars <= 5
        ):
            raise ValueError("Invalid return-anchor parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["return"] = (
            data["close"] / out["previous_close"].where(out["previous_close"] > 0.0) - 1.0
        ).replace([np.inf, -np.inf], np.nan).where(data["close"] > 0.0)
        r = out["return"]
        w = params.anchor_window
        out["gain_count"] = r.gt(0).astype(float).where(r.notna()).rolling(w).sum().shift(1)
        out["loss_count"] = r.lt(0).astype(float).where(r.notna()).rolling(w).sum().shift(1)
        out["gain_sum"] = r.clip(lower=0.0).rolling(w).sum().shift(1)
        out["loss_sum"] = r.clip(upper=0.0).rolling(w).sum().shift(1)
        out["gain_anchor"] = out["gain_sum"] / out["gain_count"].replace(0.0, np.nan)
        out["loss_anchor"] = out["loss_sum"] / out["loss_count"].replace(0.0, np.nan)
        out["recent_return"] = r.rolling(params.displacement_window).mean()
        out["anchor_span"] = out["gain_anchor"] - out["loss_anchor"]
        out["relative_position"] = (
            (out["recent_return"] - out["loss_anchor"])
            / out["anchor_span"].where(out["anchor_span"] > 0.0)
        ).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - out["previous_close"]).abs(),
                (data["low"] - out["previous_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        ready = (
            np.isfinite(out["relative_position"])
            & np.isfinite(r)
            & np.isfinite(out["atr"])
            & out["atr"].gt(0.0)
        )
        out["candidate"] = np.select(
            [
                ready & out["relative_position"].lt(params.entry_band) & r.gt(0.0),
                ready & out["relative_position"].gt(1.0 - params.entry_band) & r.lt(0.0),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ReturnAnchorParams,
    ) -> SignalFrame:
        candidate = indicators["candidate"].fillna(0).astype(int)
        entries = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_atr = np.nan
        fixed_stop = np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    # The prior close's decision fills at today's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        side = 0
                        continue
                    fixed_stop = opens[i] - side * params.stop_atr * entry_atr
                stopped = not np.isfinite(closes[i]) or side * (closes[i] - fixed_stop) < 0.0
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    side = 0
                # An exit always creates a flat next bar; no same-close reversal.
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                side = int(entries[i])
                entry_bar = i + 1
                entry_atr = atr[i]
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = ReturnAnchorParams()
    index = pd.date_range("2020-01-01", periods=70, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    computed = strategy.indicators(data, params)
    assert computed.index.equals(data.index)
    assert strategy.generate_signals(data, computed, None, params).data["signal"].eq(0).all()
    sample = data.iloc[:9].copy()
    for side in (1, -1):
        ind = pd.DataFrame({"candidate": [side, side, 0, 0, 0, 0, 0, 0, 0], "atr": 1.0}, index=sample.index)
        held = strategy.generate_signals(sample, ind, None, params).data["signal"].tolist()
        assert held == [0, 0, side, side, side, side, 0, 0, 0]
        adverse = sample.copy()
        adverse.iloc[3, adverse.columns.get_loc("close")] = 100.0 - side * 2.0
        stopped = strategy.generate_signals(adverse, ind, None, params).data["signal"].tolist()
        assert stopped == [0, 0, side, side, 0, 0, 0, 0, 0]
        ind.iloc[1, ind.columns.get_loc("candidate")] = 0
        assert strategy.generate_signals(sample, ind, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
