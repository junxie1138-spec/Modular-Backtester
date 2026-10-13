from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSlopeParams:
    slope_window: int = 12
    innovation_lag: int = 3
    atr_window: int = 14
    entry_threshold: float = 1.0
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    hold_bars: int = 18
    min_size: float = 0.25
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[GapSlopeParams]):
    strategy_id = "gen_local-w2_1791271640"

    @classmethod
    def params_type(cls):
        return GapSlopeParams

    def warmup_bars(self, params: GapSlopeParams) -> int:
        return int(max(params.slope_window + params.innovation_lag + 1, params.atr_window))

    def indicators(self, data: pd.DataFrame, params: GapSlopeParams) -> pd.DataFrame:
        if (params.slope_window < 2 or params.innovation_lag < 1
                or params.atr_window < 1 or params.hold_bars < 1
                or not 0 < params.min_size <= params.max_size <= 1
                or params.entry_threshold <= 0 or params.breakeven_pct <= 0
                or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["intraday_return"] = np.log(close / opening)
        out["gap_return"] = np.log(opening / close.shift(1))
        w = params.slope_window
        # Mean log returns estimate latent slopes of session-separated price paths.
        out["intraday_slope"] = out["intraday_return"].rolling(w, min_periods=w).mean()
        out["gap_slope"] = out["gap_return"].rolling(w, min_periods=w).mean()
        out["intraday_noise"] = out["intraday_return"].rolling(w, min_periods=w).std(ddof=0)
        out["gap_noise"] = out["gap_return"].rolling(w, min_periods=w).std(ddof=0)
        out["slope_innovation"] = out["intraday_slope"] - out["intraday_slope"].shift(params.innovation_lag)
        out["innovation_strength"] = out["slope_innovation"] * np.sqrt(w) / out["intraday_noise"].replace(0, np.nan)
        out["gap_tension"] = out["gap_slope"] * np.sqrt(w) / out["gap_noise"].replace(0, np.nan)
        out["score"] = out["innovation_strength"] * out["gap_tension"].abs().clip(upper=3.0)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["eligible"] = (
            (out["intraday_slope"] * out["slope_innovation"] > 0)
            & (out["intraday_slope"] * out["gap_slope"] < 0)
            & (out["gap_tension"].abs() >= 0.5)
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapSlopeParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        openings = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        scores = indicators["score"].to_numpy(dtype=float)
        eligible = indicators["eligible"].fillna(0).to_numpy(dtype=int)
        side, age = 0, 0
        entry = np.nan
        entry_q = peak_q = stop_q = 0.0
        armed = False
        position_size = params.min_size
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if side:
                # A prior close decision fills at this bar's open.
                if np.isnan(entry):
                    if not np.isfinite(openings[i]) or openings[i] <= 0:
                        side = 0
                        continue
                    entry = openings[i]
                    entry_q = peak_q = side * entry
                    stop_q = entry_q - params.trail_atr * atrs[i - 1]
                    armed = False
                    age = 0
                age += 1
                exit_now = age >= params.hold_bars
                if np.isfinite(closes[i]) and closes[i] > 0:
                    q = side * closes[i]
                    # Check the previously established stop before raising it.
                    exit_now = exit_now or q <= stop_q
                    peak_q = max(peak_q, q)
                    armed = armed or peak_q - entry_q >= params.breakeven_pct * entry
                    if armed:
                        # Signed stop only rises; a short's actual price stop falls.
                        stop_q = max(stop_q, entry_q)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop_q = max(stop_q, peak_q - params.trail_atr * atrs[i])
                        exit_now = exit_now or q <= stop_q
                if exit_now:
                    side = 0
                    entry = np.nan
                else:
                    decisions[i] = side
                    sizes[i] = position_size
            elif (i >= warmup - 1 and eligible[i]
                    and np.isfinite(scores[i])
                    and np.isfinite(atrs[i]) and atrs[i] > 0
                    and abs(scores[i]) >= params.entry_threshold):
                side = 1 if scores[i] > 0 else -1
                entry = np.nan
                strength = min(abs(scores[i]) / (3.0 * params.entry_threshold), 1.0)
                position_size = params.min_size + (params.max_size - params.min_size) * strength
                decisions[i] = side
                sizes[i] = position_size
        df["signal"] = decisions
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GapSlopeParams(slope_window=2, innovation_lag=1, atr_window=2,
                            breakeven_pct=0.02, trail_atr=1.0)
    index = pd.date_range("2020-01-01", periods=16)
    close = np.full(16, 100.0)
    close[5:9] = [101.0, 103.0, 102.0, 100.5]
    close[11:15] = [99.0, 97.0, 98.0, 99.5]
    data = pd.DataFrame({"open": 100.0, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 1000}, index=index)
    ind = pd.DataFrame({"atr": 2.0, "score": 0.0, "eligible": 0}, index=index)
    ind.iloc[4, ind.columns.get_loc("score")] = 2.0
    ind.iloc[10, ind.columns.get_loc("score")] = -2.0
    ind.iloc[[4, 10], ind.columns.get_loc("eligible")] = 1
    ind.iloc[[7, 13], ind.columns.get_loc("atr")] = 10.0
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=4)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 0]
    assert result["size"].gt(0).all() and result["size"].iloc[5] == 0.75
    real_ind = strategy.indicators(data, params)
    assert real_ind.index.equals(data.index)
    real_result = strategy.generate_signals(data, real_ind, ctx, params).data
    assert real_result["signal"].isin([-1, 0, 1]).all()
    prefix = strategy.generate_signals(data.iloc[:12], real_ind.iloc[:12], ctx, params).data
    pd.testing.assert_frame_equal(prefix, real_result.iloc[:12])


if __name__ == "__main__":
    _self_check()
