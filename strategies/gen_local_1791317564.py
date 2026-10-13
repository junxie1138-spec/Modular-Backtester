from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    streak_bars: int = 3
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791317564"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(50, params.streak_bars + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.streak_bars < 1 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("streak_bars must be positive and trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["volatility"] = out["return"].rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252.0)
        out["trend"] = data["close"].rolling(50, min_periods=50).mean()
        out["contracting"] = out["true_range"].lt(out["true_range"].shift(1))
        groups = (~out["contracting"]).cumsum()
        out["streak"] = out["contracting"].astype(int).groupby(groups).cumsum()
        out["ready"] = (
            np.isfinite(out["atr"]) & out["atr"].gt(0)
            & np.isfinite(out["volatility"]) & out["volatility"].gt(0)
            & np.isfinite(out["trend"])
        )
        out["entry"] = (
            out["ready"]
            & out["streak"].shift(1).ge(params.streak_bars)
            & out["true_range"].gt(out["true_range"].shift(1))
            & data["close"].gt(data["high"].shift(1))
            & data["close"].gt(out["trend"])
        ).fillna(False)
        out["target_size"] = (
            0.12 / out["volatility"].where(out["volatility"].gt(0))
        ).replace([np.inf, -np.inf], np.nan).clip(0.05, 0.95).fillna(0.05)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["ready"].to_numpy(dtype=bool)
        entries = indicators["entry"].to_numpy(dtype=bool)
        active = False
        pending_entry = False
        armed = False
        entry = stop = peak = 0.0
        age = 0

        for i in range(len(data)):
            valid_bar = (
                np.isfinite(opens[i]) and np.isfinite(highs[i])
                and np.isfinite(lows[i]) and np.isfinite(closes[i])
                and opens[i] > 0 and closes[i] > 0
                and lows[i] > 0 and highs[i] >= lows[i]
            )
            if pending_entry:
                pending_entry = False
                if valid_bar and i > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    # The prior close's decision fills at this bar's open.
                    entry = opens[i]
                    stop = entry - params.trail_atr * atr[i - 1]
                    peak = entry
                    age = 0
                    armed = False
                    active = True

            if active:
                age += 1
                if not valid_bar or not ready[i]:
                    active = False
                elif lows[i] <= stop or age >= 18:
                    # Stop observations request an exit at the next open.
                    active = False
                else:
                    peak = max(peak, highs[i])
                    if peak >= entry * 1.02:
                        armed = True
                    if armed:
                        # A newly raised stop applies prospectively; only the
                        # current close is compared with it on this bar.
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                        if closes[i] <= stop:
                            active = False
                decisions[i] = int(active)
            elif valid_bar and entries[i]:
                decisions[i] = 1
                pending_entry = True

        df["signal"] = decisions
        df["size"] = indicators["target_size"].astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=80, freq="B")
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 10000.0,
    }, index=index)
    for i, width in zip(range(48, 52), [6.0, 5.0, 4.0, 3.0]):
        data.iloc[i, data.columns.get_loc("high")] = 100.0 + width / 2
        data.iloc[i, data.columns.get_loc("low")] = 100.0 - width / 2
    data.iloc[52] = [100.0, 105.0, 99.0, 104.0, 10000.0]
    data.iloc[53] = [104.0, 108.0, 103.0, 107.0, 10000.0]
    data.iloc[54] = [107.0, 107.0, 100.0, 101.0, 10000.0]
    strategy = GeneratedStrategy()
    params = StrategyParams()
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert indicators["streak"].iloc[51] == 3
    assert indicators["entry"].iloc[52]
    assert result["signal"].iloc[52:56].tolist() == [0, 1, 1, 0]
    assert set(result["signal"].unique()) <= {0, 1}
    assert np.isfinite(result["size"]).all() and result["size"].gt(0).all()
    changed = data.copy()
    changed.iloc[56:, :4] *= 1.5
    changed_result = strategy.generate_signals(
        changed, strategy.indicators(changed, params), None, params
    ).data
    pd.testing.assert_frame_equal(result.iloc[:56], changed_result.iloc[:56])


if __name__ == "__main__":
    _self_check()
