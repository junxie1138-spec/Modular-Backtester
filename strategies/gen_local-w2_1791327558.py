from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    rank_window: int = 63
    shape_bars: int = 6
    history_bars: int = 252
    outcome_bars: int = 8
    hold_bars: int = 10
    atr_bars: int = 14
    min_matches: int = 8
    max_distance: float = 0.18
    min_edge: float = 0.002
    min_win_rate: float = 0.60
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local-w2_1791327558"

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return max(200, params.atr_bars + 1,
                   params.rank_window + 2 * params.shape_bars
                   + params.outcome_bars + params.history_bars)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        p = params
        if min(p.rank_window, p.shape_bars, p.history_bars, p.outcome_bars,
               p.hold_bars, p.atr_bars, p.min_matches) < 1:
            raise ValueError("Integer parameters must be positive.")
        if not (0 < p.max_distance <= 1 and 0 <= p.min_win_rate <= 1
                and p.stop_atr > 0 and np.isfinite(p.stop_atr)
                and np.isfinite(p.min_edge)):
            raise ValueError("Invalid distance, probability, edge, or ATR multiplier.")
        close = data["close"].where(data["close"] > 0)
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        returns = close.pct_change(fill_method=None)
        out["return_rank"] = returns.rolling(p.rank_window).rank(pct=True)
        out["volume_rank"] = data["volume"].where(data["volume"] >= 0).rolling(
            p.rank_window).rank(pct=True)
        previous = close.shift(1)
        tr = pd.concat([data["high"] - data["low"],
                        (data["high"] - previous).abs(),
                        (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = tr.rolling(p.atr_bars).mean().where(lambda s: s > 0)
        columns = []
        for lag in range(p.shape_bars):
            for name in ("return_rank", "volume_rank"):
                column = f"{name}_{lag}"
                out[column] = out[name].shift(lag)
                columns.append(column)
        features = out[columns]
        out["matured_return"] = close.pct_change(p.outcome_bars, fill_method=None)
        count = np.zeros(len(data), dtype=float)
        total = np.zeros(len(data), dtype=float)
        wins = np.zeros(len(data), dtype=float)
        # Candidate outcomes finish before the live shape starts.
        first_lag = p.outcome_bars + p.shape_bars
        for lag in range(first_lag, first_lag + p.history_bars):
            distance = (features - features.shift(lag)).abs().mean(axis=1, skipna=False)
            outcome = out["matured_return"].shift(lag - p.outcome_bars)
            eligible = (distance.le(p.max_distance) & outcome.notna()
                        & close.shift(lag).gt(out["ma200"].shift(lag)))
            count += eligible.to_numpy(dtype=float)
            total += outcome.where(eligible, 0.0).to_numpy()
            wins += (eligible & outcome.gt(0)).to_numpy(dtype=float)
        out["analog_count"] = count
        denominator = np.where(count > 0, count, np.nan)
        out["analog_mean"] = total / denominator
        out["analog_win_rate"] = wins / denominator
        out["shock_cleared"] = (
            out["volume_rank"].shift(p.shape_bars - 1).gt(0.80)
            & out["volume_rank"].lt(0.50)
            & out["return_rank"].gt(0.50)
        )
        out["ready"] = (np.arange(len(data)) >= self.warmup_bars(p) - 1)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        p = params
        entry = (
            indicators["ready"]
            & indicators["shock_cleared"]
            & data["close"].gt(indicators["ma200"])
            & indicators["analog_count"].ge(p.min_matches)
            & indicators["analog_mean"].ge(p.min_edge)
            & indicators["analog_win_rate"].ge(p.min_win_rate)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        regimes = data["close"].gt(indicators["ma200"]).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0
            if active:
                age += 1
                if valid:
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - p.stop_atr * atrs[i])
                if not valid or not regimes[i] or closes[i] <= stop or age >= p.hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif valid and entry[i]:
                active = True
                age = 0
                # The first held close is next bar, after the entry fill.
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    p = ShapeParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 104., 103., 99., 103., 104.]}, index=index)
    derived = pd.DataFrame({
        "ready": True, "shock_cleared": True, "ma200": 90.,
        "analog_count": 20., "analog_mean": 0.01, "analog_win_rate": 0.8,
        "atr": [2., 2., 5., 5., 2., 2.]
    }, index=index)
    result = strategy.generate_signals(data, derived, None, p).data
    # ATR expansion must not lower the stop from 100; exit at 99 fills next bar.
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 1]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
