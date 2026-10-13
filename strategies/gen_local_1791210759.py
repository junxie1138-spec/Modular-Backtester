from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilitySeasonParams:
    graph_bars: int = 12
    degree_gap: int = 3
    seasonal_bars: int = 252
    min_season_samples: int = 24
    seasonal_edge: float = 0.001
    atr_bars: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VisibilitySeasonParams]):
    strategy_id = "gen_local_1791210759"

    @classmethod
    def params_type(cls):
        return VisibilitySeasonParams

    def warmup_bars(self, params: VisibilitySeasonParams) -> int:
        return int(max(params.graph_bars + 1, params.seasonal_bars + 5, params.atr_bars + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: VisibilitySeasonParams) -> pd.DataFrame:
        if not (params.graph_bars >= 2 and params.degree_gap >= 1
                and params.seasonal_bars >= 5 and params.min_season_samples >= 1
                and params.atr_bars >= 1 and 3 <= params.hold_bars <= 5
                and params.seasonal_edge >= 0 and params.breakeven_pct > 0
                and params.trail_atr > 0):
            raise ValueError("Invalid visibility-seasonality parameters")
        out = pd.DataFrame(index=data.index)
        log_close = np.log(data["close"].where(data["close"] > 0))
        out["log_close"] = log_close
        upper = np.zeros(len(data), dtype=float)
        lower = np.zeros(len(data), dtype=float)
        maximum = np.full(len(data), -np.inf)
        minimum = np.full(len(data), np.inf)
        # Natural visibility: a past vertex is linked if no nearer slope blocks it.
        for lag in range(1, params.graph_bars + 1):
            slope = ((log_close.shift(lag) - log_close) / lag).to_numpy()
            upper += (slope > maximum)
            lower += (slope < minimum)
            maximum = np.maximum(maximum, slope)
            minimum = np.minimum(minimum, slope)
        valid_graph = log_close.rolling(params.graph_bars + 1).count().eq(params.graph_bars + 1)
        out["upper_degree"] = pd.Series(upper, index=data.index).where(valid_graph)
        out["lower_degree"] = pd.Series(lower, index=data.index).where(valid_graph)
        out["topology"] = out["lower_degree"] - out["upper_degree"]
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        out["weekday"] = weekday
        # At t, open[t]/open[t-4] is the matured outcome of decision t-5.
        out["matured_return"] = np.log(data["open"].where(data["open"] > 0)).diff(4)
        out["origin_weekday"] = weekday.shift(5)
        seasonal = pd.Series(np.nan, index=data.index, dtype=float)
        for day in range(5):
            observed = out["matured_return"].where(out["origin_weekday"].eq(day))
            count = observed.notna().astype(float).rolling(
                params.seasonal_bars, min_periods=params.seasonal_bars
            ).sum()
            total = observed.fillna(0.0).rolling(
                params.seasonal_bars, min_periods=params.seasonal_bars
            ).sum()
            estimate = (total / count.replace(0.0, np.nan)).where(count >= params.min_season_samples)
            out[f"seasonal_{day}"] = estimate
            seasonal = seasonal.where(~weekday.eq(day), estimate)
        out["seasonal_edge"] = seasonal
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: VisibilitySeasonParams,
    ) -> SignalFrame:
        valid = (
            np.isfinite(indicators["topology"])
            & np.isfinite(indicators["seasonal_edge"])
            & np.isfinite(indicators["atr"])
            & indicators["atr"].gt(0)
        )
        raw = pd.Series(0, index=data.index, dtype=int)
        raw.loc[valid & indicators["topology"].ge(params.degree_gap)
                & indicators["seasonal_edge"].ge(params.seasonal_edge)] = 1
        raw.loc[valid & indicators["topology"].le(-params.degree_gap)
                & indicators["seasonal_edge"].le(-params.seasonal_edge)] = -1
        confirmed = raw.where(raw.eq(raw.shift(1)) & raw.ne(0), 0).to_numpy()
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        side = 0
        pending = False
        armed = False
        entry = best = stop = 0.0
        age = 0
        for i in range(len(data)):
            if side:
                if pending:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        pending = False
                        continue
                    entry = opens[i]
                    best = side * entry
                    stop = best - params.trail_atr * atr[i - 1]
                    pending = False
                    armed = False
                    age = 0
                age += 1
                if not np.isfinite([highs[i], lows[i], closes[i], atr[i]]).all():
                    side = 0
                    continue
                adverse = lows[i] if side == 1 else -highs[i]
                favorable = highs[i] if side == 1 else -lows[i]
                # Stops ratchet upward in directional price coordinates for both sides.
                exit_now = adverse <= stop
                best = max(best, favorable)
                if best - side * entry >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, side * entry, best - params.trail_atr * atr[i])
                    # New intrabar highs/lows only tighten the close-tested stop.
                    exit_now = exit_now or side * closes[i] <= stop
                if exit_now or age >= params.hold_bars:
                    side = 0
                    continue
                decisions[i] = side
            elif confirmed[i] and valid.iloc[i]:
                side = int(confirmed[i])
                pending = True
                decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = VisibilitySeasonParams()
    index = pd.bdate_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 100.1, "low": 99.9,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        ind = pd.DataFrame({"topology": 0.0, "seasonal_edge": direction * 0.01,
                            "atr": 1.0}, index=index)
        ind.iloc[:2, 0] = direction * params.degree_gap
        signal = strategy.generate_signals(data, ind, ctx, params).data["signal"]
        assert signal.tolist() == [0, 0, direction, direction, direction, direction, 0, 0, 0, 0, 0, 0]
        stopped = data.copy()
        stopped.iloc[2, stopped.columns.get_loc("high")] = 101.0 if direction == 1 else 100.1
        stopped.iloc[2, stopped.columns.get_loc("low")] = 99.0 if direction == -1 else 99.9
        signal = strategy.generate_signals(stopped, ind, ctx, params).data["signal"]
        assert signal.iloc[2] == direction and signal.iloc[3:].eq(0).all()
