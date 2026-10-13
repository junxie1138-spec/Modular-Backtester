from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapShiftParams:
    baseline_bars: int = 20
    recent_bars: int = 3
    atr_bars: int = 14
    gap_z: float = 1.0
    trail_atr: float = 1.5
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[GapShiftParams]):
    strategy_id = "gen_local-w2_1791335780"

    @classmethod
    def params_type(cls):
        return GapShiftParams

    def warmup_bars(self, params: GapShiftParams) -> int:
        return max(params.baseline_bars + params.recent_bars + 3, params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: GapShiftParams) -> pd.DataFrame:
        if min(params.baseline_bars, params.recent_bars, params.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 3 <= params.max_hold <= 5:
            raise ValueError("max_hold must be between three and five sessions.")
        if not np.isfinite([params.gap_z, params.trail_atr]).all() or min(params.gap_z, params.trail_atr) <= 0:
            raise ValueError("Gap threshold and ATR multiplier must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["gap"] = (data["open"] / out["prior_close"].where(out["prior_close"] > 0) - 1).replace([np.inf, -np.inf], np.nan)
        out["recent_gap"] = out["gap"].rolling(params.recent_bars, min_periods=params.recent_bars).mean()
        old_gap = out["gap"].shift(params.recent_bars)
        out["baseline_gap"] = old_gap.rolling(params.baseline_bars, min_periods=params.baseline_bars).mean()
        out["baseline_noise"] = old_gap.rolling(params.baseline_bars, min_periods=params.baseline_bars).std(ddof=0)
        out["gap_shift"] = (out["recent_gap"] - out["baseline_gap"]) / out["baseline_noise"].clip(lower=1e-6)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["qualified"] = (
            (out["baseline_gap"] < 0)
            & (out["recent_gap"] > 0)
            & (out["gap_shift"] >= params.gap_z)
            & (data["close"] > data["open"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapShiftParams,
    ) -> SignalFrame:
        qualified = indicators["qualified"].fillna(False).astype(bool)
        confirmed = qualified & qualified.shift(1, fill_value=False)
        entry = (confirmed & ~confirmed.shift(1, fill_value=False)).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        entry_bar = -1
        highest_close = stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if entry_bar >= 0:
                if not valid:
                    entry_bar = -1
                else:
                    # First in-trade close is on the next bar, after the entry fill.
                    highest_close = max(highest_close, closes[i])
                    stop = max(stop, highest_close - params.trail_atr * atr[i])
                    if closes[i] <= stop or i - entry_bar + 1 >= params.max_hold:
                        entry_bar = -1
            elif entry[i] and valid:
                entry_bar = i + 1
                highest_close = stop = -np.inf
            raw[i] = int(entry_bar >= 0)
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapShiftParams()
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    close = pd.Series([100., 100., 102., 101., 99., 99., 99., 99.], index=index)
    data = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1000.}, index=index)
    ind = pd.DataFrame({"qualified": [True, True, False, False, False, False, False, False], "atr": [1., 1., 1., 10., 10., 1., 1., 1.]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
    ind.loc[index[0], "qualified"] = False
    assert not strategy.generate_signals(data, ind, ctx, params).data["signal"].any()
    ind["qualified"] = [True, True, False, False, False, False, False, False]
    ind["atr"] = 100.
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert not strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
