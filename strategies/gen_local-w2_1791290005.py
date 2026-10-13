from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_bars: int = 3
    analog_bars: int = 100
    neighbors: int = 9
    atr_bars: int = 14
    volume_bars: int = 20
    hold_bars: int = 4
    forecast_min: float = 0.002
    volume_min: float = 1.05
    win_min: float = 0.66
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    initial_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791290005"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(
            params.hold_bars + params.analog_bars + 2 * params.shape_bars - 2
            + max(params.atr_bars + 1, params.volume_bars + 1)
        )

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        if (
            min(p.shape_bars, p.analog_bars, p.neighbors, p.atr_bars, p.volume_bars) < 1
            or p.neighbors > p.analog_bars or not 3 <= p.hold_bars <= 5
            or not 0 <= p.win_min <= 1 or p.forecast_min < 0
            or min(p.volume_min, p.breakeven_pct, p.trail_atr, p.initial_atr) <= 0
        ):
            raise ValueError("Invalid analog or risk parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        out["return"] = close.div(previous).sub(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_bars, min_periods=p.atr_bars).mean()
        out["volume_reference"] = data["volume"].shift(1).rolling(
            p.volume_bars, min_periods=p.volume_bars
        ).mean()
        out["volume_ratio"] = data["volume"].where(data["volume"] > 0).div(
            out["volume_reference"].where(out["volume_reference"] > 0)
        )
        out["price_step"] = close.sub(previous).div(
            out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        ).clip(-4, 4)
        out["volume_step"] = np.log(out["volume_ratio"]).clip(-2, 2)
        shape = pd.concat(
            [out[column].shift(j).rename(column + "_" + str(j))
             for column in ("price_step", "volume_step") for j in range(p.shape_bars)],
            axis=1
        )
        out = pd.concat([out, shape], axis=1)
        current = shape.to_numpy(dtype=float)
        distances = []
        outcomes = []
        # Mature analog outcomes end before the current shape begins.
        for lag in range(p.hold_bars + p.shape_bars,
                         p.hold_bars + p.shape_bars + p.analog_bars):
            historical = shape.shift(lag).to_numpy(dtype=float)
            distance = np.square(current - historical).mean(axis=1)
            outcome = close.shift(lag - p.hold_bars).div(close.shift(lag)).sub(1)
            values = outcome.to_numpy(dtype=float)
            distances.append(np.where(
                np.isfinite(distance) & np.isfinite(values), distance, np.inf
            ))
            outcomes.append(values)
        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        nearest = np.argsort(distance_matrix, axis=1, kind="stable")[:, :p.neighbors]
        selected_distances = np.take_along_axis(distance_matrix, nearest, axis=1)
        selected_outcomes = np.take_along_axis(outcome_matrix, nearest, axis=1)
        ready = np.isfinite(selected_distances).all(axis=1)
        out["analog_forecast"] = np.where(ready, selected_outcomes.mean(axis=1), np.nan)
        out["analog_win_rate"] = np.where(
            ready, (selected_outcomes > 0).mean(axis=1), np.nan
        )
        out["analog_distance"] = np.where(ready, selected_distances.mean(axis=1), np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams
    ) -> SignalFrame:
        p = params
        qualifying = (
            (indicators["return"] > 0)
            & (indicators["volume_ratio"] >= p.volume_min)
            & (indicators["analog_forecast"] >= p.forecast_min)
            & (indicators["analog_win_rate"] >= p.win_min)
            & (indicators["atr"] > 0)
        ).fillna(False)
        qualifying.iloc[:max(0, self.warmup_bars(p) - 1)] = False
        confirmed = (qualifying & qualifying.shift(1, fill_value=False)).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decision = np.zeros(len(data), dtype=int)
        active = False
        armed = False
        entry = peak = stop = 0.0
        age = 0
        for i in range(len(data)):
            # The previous close's entry decision fills at today's open.
            if not active and i > 0 and decision[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]):
                    active = True
                    armed = False
                    entry = peak = opens[i]
                    stop = entry - p.initial_atr * atr[i - 1]
                    age = 0
            if active:
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                    continue
                # Close-confirmed stops; exit orders fill at the following open.
                peak = max(peak, closes[i])
                armed = armed or peak >= entry * (1 + p.breakeven_pct)
                if armed:
                    stop = max(stop, entry, peak - p.trail_atr * atr[i])
                if closes[i] <= stop or age >= p.hold_bars:
                    active = False
                else:
                    decision[i] = 1
            elif confirmed[i]:
                decision[i] = 1
        df = pd.DataFrame({"signal": decision, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    p = AnalogParams(shape_bars=2, analog_bars=3, neighbors=2, atr_bars=2, volume_bars=2)
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.1, "volume": 1000.0},
        index=index
    )
    indicator = pd.DataFrame(
        {"return": 0.01, "volume_ratio": 2.0, "analog_forecast": 0.01,
         "analog_win_rate": 1.0, "atr": 1.0}, index=index
    )
    start = strategy.warmup_bars(p) - 1
    indicator.iloc[start + 2:, indicator.columns.get_loc("analog_forecast")] = 0.0
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(p))
    signal = strategy.generate_signals(data, indicator, ctx, p).data["signal"]
    assert signal.iloc[start + 1] == 0 and signal.iloc[start + 2] == 1
    assert signal.iloc[start + 5] == 1 and signal.iloc[start + 6] == 0
    data.iloc[start + 2:start + 5, data.columns.get_loc("close")] = [101.0, 103.0, 101.0]
    signal = strategy.generate_signals(data, indicator, ctx, p).data["signal"]
    assert signal.iloc[start + 4] == 1 and signal.iloc[start + 5] == 0
    full = strategy.indicators(data, p)
    prefix = strategy.indicators(data.iloc[:17], p)
    assert np.allclose(full.iloc[:17].to_numpy(), prefix.to_numpy(), equal_nan=True)


if __name__ == "__main__":
    _self_check()
