from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    rank_window: int = 63
    shape_bars: int = 6
    analog_lookback: int = 252
    forecast_rank_window: int = 126
    hold_bars: int = 18
    match_distance: float = 0.30
    entry_percentile: float = 0.80
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791280176"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(200, 15, params.rank_window + params.shape_bars
                       + params.analog_lookback + params.forecast_rank_window))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if (params.rank_window < 2 or params.shape_bars < 2
                or params.forecast_rank_window < 2 or params.hold_bars < 1
                or params.analog_lookback < params.hold_bars + params.shape_bars
                or not 0.0 < params.match_distance <= 1.0
                or not 0.5 < params.entry_percentile < 1.0
                or params.breakeven_pct <= 0.0 or params.trail_atr <= 0.0):
            raise ValueError("Invalid analog or exit parameters.")
        close = data["close"].where(data["close"] > 0.0)
        span = (data["high"] - data["low"]).where(lambda x: x > 0.0)
        lower_wick = (pd.concat([data["open"], close], axis=1).min(axis=1)
                      - data["low"]).clip(lower=0.0)
        upper_wick = (data["high"]
                      - pd.concat([data["open"], close], axis=1).max(axis=1)).clip(lower=0.0)
        wick_balance = ((lower_wick - upper_wick) / span).clip(-1.0, 1.0)
        previous = close.shift(1)
        true_range = pd.concat([
            span, (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        wick_rank = wick_balance.rolling(params.rank_window).rank(pct=True)
        volume_rank = data["volume"].where(data["volume"] >= 0.0).rolling(
            params.rank_window
        ).rank(pct=True)
        columns = {
            "range": span, "lower_wick": lower_wick,
            "upper_wick": upper_wick, "wick_balance": wick_balance,
            "wick_rank": wick_rank, "volume_rank": volume_rank,
            "true_range": true_range, "atr": true_range.rolling(14).mean(),
            "ma200": close.rolling(200).mean()
        }
        weighted_sum = pd.Series(0.0, index=data.index)
        total_weight = pd.Series(0.0, index=data.index)
        match_count = pd.Series(0, index=data.index, dtype=int)
        last_lag = 0
        # Archive outcomes finish before the current footprint begins.
        for lag in range(params.hold_bars + params.shape_bars,
                         params.analog_lookback + 1, params.shape_bars):
            distance = pd.Series(0.0, index=data.index)
            for offset in range(params.shape_bars):
                distance += (wick_rank.shift(offset)
                             - wick_rank.shift(lag + offset)).abs()
                distance += (volume_rank.shift(offset)
                             - volume_rank.shift(lag + offset)).abs()
            distance /= 2.0 * params.shape_bars
            outcome = close.shift(lag - params.hold_bars) / close.shift(lag) - 1.0
            weight = (1.0 - distance / params.match_distance).clip(lower=0.0) ** 2
            weight = weight.where(outcome.notna(), 0.0).fillna(0.0)
            columns[f"distance_{lag}"] = distance
            columns[f"outcome_{lag}"] = outcome
            columns[f"weight_{lag}"] = weight
            weighted_sum += weight * outcome.fillna(0.0)
            total_weight += weight
            match_count += (weight > 0.0).astype(int)
            last_lag = lag
        archive_ready = (
            wick_rank.shift(last_lag + params.shape_bars - 1).notna()
            & volume_rank.shift(last_lag + params.shape_bars - 1).notna()
        )
        forecast = (weighted_sum / total_weight.replace(0.0, np.nan)).where(
            archive_ready & (match_count >= 3)
        )
        # Partial windows permit occasional missing analog forecasts; warmup
        # still reserves the complete chronological ranking window.
        forecast_rank = forecast.rolling(
            params.forecast_rank_window,
            min_periods=max(2, params.forecast_rank_window // 2)
        ).rank(pct=True)
        columns.update({
            "match_count": match_count, "total_weight": total_weight,
            "weighted_outcome": weighted_sum, "forecast": forecast,
            "forecast_rank": forecast_rank
        })
        out = pd.DataFrame(columns, index=data.index).replace(
            [np.inf, -np.inf], np.nan
        )
        valid = out[["atr", "ma200", "forecast", "forecast_rank"]].notna().all(axis=1)
        valid &= out["atr"] > 0.0
        long_entry = (valid & (close > out["ma200"]) & (forecast > 0.0)
                      & (forecast_rank >= params.entry_percentile))
        short_entry = (valid & (close < out["ma200"]) & (forecast < 0.0)
                       & (forecast_rank <= 1.0 - params.entry_percentile))
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        commands = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        position, entry_bar = 0, -1
        entry_price, stop, best = np.nan, np.nan, np.nan
        armed = False
        for i in range(self.warmup_bars(params) - 1, len(data)):
            opened, high, low, closed = prices[i]
            if not np.isfinite(prices[i]).all() or min(prices[i]) <= 0.0:
                position = 0
                continue
            if position:
                if i == entry_bar:
                    # The prior close's command fills at this bar's open.
                    entry_price = opened
                    best = position * entry_price
                    stop = best - params.trail_atr * atr[i - 1]
                    armed = False
                # Signed-price stops only increase; short raw-price stops fall.
                adverse = low if position == 1 else -high
                if (not np.isfinite(atr[i]) or atr[i] <= 0.0
                        or adverse <= stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    position = 0
                else:
                    favorable = position * closed
                    best = max(best, favorable)
                    if position * (closed / entry_price - 1.0) >= params.breakeven_pct:
                        armed = True
                    if armed:
                        stop = max(stop, position * entry_price,
                                   best - params.trail_atr * atr[i])
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                position = int(entries[i])
                entry_bar = i + 1
            commands[i] = position
        df["signal"] = commands
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = AnalogParams()
    rng = np.random.default_rng(17)
    index = pd.date_range("2000-01-01", periods=950, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(index))))
    opened = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0.0, 0.004, len(index)))
    data = pd.DataFrame({
        "open": opened, "high": np.maximum(opened, close) * 1.01,
        "low": np.minimum(opened, close) * 0.99, "close": close,
        "volume": rng.integers(100000, 1000000, len(index))
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    full = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, full, ctx, params).data
    prefix = data.iloc[:800]
    partial = strategy.indicators(prefix, params)
    pd.testing.assert_frame_equal(full.iloc[:800], partial)
    pd.testing.assert_frame_equal(
        signals.iloc[:800], strategy.generate_signals(prefix, partial, ctx, params).data
    )
    assert signals["signal"].isin([-1, 0, 1]).all()
    assert (signals["size"] > 0.0).all()
    assert (signals.iloc[:strategy.warmup_bars(params)]["signal"] == 0).all()
    # Check delayed fills and breakeven exits in both directions.
    for direction in (1, -1):
        sample = pd.DataFrame(100.0, index=index[:230],
                              columns=["open", "high", "low", "close", "volume"])
        small = AnalogParams(rank_window=2, shape_bars=2,
                             analog_lookback=8, forecast_rank_window=2)
        features = pd.DataFrame({"atr": 1.0, "entry": 0}, index=sample.index)
        features.iloc[200, features.columns.get_loc("entry")] = direction
        sample.iloc[201, sample.columns.get_loc("close")] = 100.0 + direction * 2.0
        sample.iloc[201, sample.columns.get_loc("high")] = 103.0 if direction == 1 else 100.5
        sample.iloc[201, sample.columns.get_loc("low")] = 99.5 if direction == 1 else 97.0
        sample.iloc[202, sample.columns.get_loc("low")] = 99.0
        sample.iloc[202, sample.columns.get_loc("high")] = 101.0
        result = strategy.generate_signals(sample, features, ctx, small).data["signal"]
        assert result.iloc[200] == 0
        assert result.iloc[201] == direction
        assert result.iloc[202] == direction
        assert result.iloc[203] == 0


if __name__ == "__main__":
    _self_check()
