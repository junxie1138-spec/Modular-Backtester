from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    fd_order: float = 0.4
    fd_window: int = 24
    shape_bars: int = 5
    history_bars: int = 240
    neighbors: int = 5
    min_edge: float = 0.002
    pressure_bars: int = 3
    min_pressure: float = 0.12
    profit_target: float = 0.012
    max_hold: int = 4
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791288804"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(
            params.fd_window + 2 * params.shape_bars
            + params.history_bars + params.max_hold,
            params.pressure_bars,
        ))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if not (
            0.0 < params.fd_order < 1.0 and params.fd_window >= 2
            and params.shape_bars >= 3 and params.history_bars >= params.max_hold
            and params.neighbors >= 1 and params.pressure_bars >= 1
            and 3 <= params.max_hold <= 5 and params.min_edge > 0
            and 0 <= params.min_pressure < 1 and params.profit_target > 0
            and params.size > 0
        ):
            raise ValueError("Invalid analog strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0).replace([np.inf, -np.inf], np.nan)
        frac = close.copy()
        weight = 1.0
        for lag in range(1, params.fd_window):
            weight *= (lag - 1 - params.fd_order) / lag
            frac = frac + weight * close.shift(lag)
        out["frac_close"] = frac

        shape = pd.concat([frac.shift(j) for j in range(params.shape_bars)], axis=1)
        scale = shape.std(axis=1, ddof=0).where(shape.notna().all(axis=1))
        shape = shape.sub(shape.mean(axis=1), axis=0).div(
            scale.where(scale > 1e-12), axis=0
        )
        for j in range(params.shape_bars):
            out[f"shape_{j}"] = shape.iloc[:, j]
        vectors = shape.to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        up_share = np.full(len(data), np.nan)
        embargo = params.shape_bars + params.max_hold + 1
        # Nonoverlapping outcome intervals; all labels mature before the query shape.
        for i in range(self.warmup_bars(params) - 1, len(data)):
            if not np.isfinite(vectors[i]).all():
                continue
            anchors = np.arange(
                i - embargo, i - embargo - params.history_bars,
                -params.max_hold, dtype=int,
            )
            anchors = anchors[anchors >= 0]
            start = opens[anchors + 1]
            finish = opens[anchors + params.max_hold + 1]
            valid = (
                np.isfinite(vectors[anchors]).all(axis=1)
                & np.isfinite(start) & np.isfinite(finish)
                & (start > 0) & (finish > 0)
            )
            anchors, start, finish = anchors[valid], start[valid], finish[valid]
            if len(anchors) < params.neighbors:
                continue
            distance = ((vectors[anchors] - vectors[i]) ** 2).mean(axis=1)
            nearest = np.argsort(distance, kind="stable")[:params.neighbors]
            outcomes = finish[nearest] / start[nearest] - 1.0
            forecast[i] = outcomes.mean()
            up_share[i] = (outcomes > 0).mean()
        out["analog_return"] = forecast
        out["analog_up_share"] = up_share

        out["candle_body"] = (data["close"] - data["open"]).replace(
            [np.inf, -np.inf], np.nan
        )
        out["candle_range"] = (data["high"] - data["low"]).where(
            data["high"] >= data["low"]
        ).replace([np.inf, -np.inf], np.nan)
        out["body_sum"] = out["candle_body"].rolling(params.pressure_bars).sum()
        out["range_sum"] = out["candle_range"].rolling(params.pressure_bars).sum()
        out["pressure"] = out["body_sum"] / out["range_sum"].where(out["range_sum"] > 0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams,
    ) -> SignalFrame:
        long_entry = (
            (indicators["analog_return"] >= params.min_edge)
            & (indicators["analog_up_share"] >= 0.8)
            & (indicators["pressure"] >= params.min_pressure)
            & (indicators["pressure"] > 0)
        )
        short_entry = (
            (indicators["analog_return"] <= -params.min_edge)
            & (indicators["analog_up_share"] <= 0.2)
            & (indicators["pressure"] <= -params.min_pressure)
            & (indicators["pressure"] < 0)
        )
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], 0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position, held = 0, 0
        entry_open = np.nan
        for i in range(len(data)):
            if position:
                if held == 0:
                    entry_open = opens[i]
                held += 1
                valid_entry = np.isfinite(entry_open) and entry_open > 0
                gain = position * (closes[i] / entry_open - 1.0) if valid_entry else np.nan
                # Close-confirmed target; the shifted zero exits at the next open.
                if not valid_entry or gain >= params.profit_target or held >= params.max_hold:
                    position, held, entry_open = 0, 0, np.nan
            elif i >= self.warmup_bars(params) - 1:
                position = int(entries[i])
                held = 0
            raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = AnalogParams()
    strategy = GeneratedStrategy()
    anchor = strategy.warmup_bars(params) - 1
    index = pd.date_range("2020-01-01", periods=anchor + 20)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    readings = pd.DataFrame(
        {"analog_return": np.nan, "analog_up_share": np.nan, "pressure": np.nan},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=anchor + 1)
    for direction in (1, -1):
        readings.iloc[anchor] = [direction * 0.01, float(direction == 1), direction * 0.5]
        signals = strategy.generate_signals(data, readings, ctx, params).data
        expected = np.zeros(len(data), dtype=int)
        expected[anchor + 1:anchor + params.max_hold + 1] = direction
        assert np.array_equal(signals["signal"].to_numpy(), expected)
        data.iloc[anchor + 1, data.columns.get_loc("close")] = 100 * (1 + direction * 0.02)
        signals = strategy.generate_signals(data, readings, ctx, params).data
        expected[anchor + 2:] = 0
        assert np.array_equal(signals["signal"].to_numpy(), expected)
        data["close"] = 100.0
    t = np.arange(len(data))
    data["open"] = 100 + 0.02 * t + np.sin(t * 0.35)
    data["close"] = data["open"] + 0.3 * np.cos(t * 0.6)
    data["high"] = data[["open", "close"]].max(axis=1) + 1
    data["low"] = data[["open", "close"]].min(axis=1) - 1
    full = strategy.indicators(data, params)
    cut = len(data) - 7
    prefix = strategy.indicators(data.iloc[:cut], params)
    assert np.allclose(full.iloc[:cut].to_numpy(), prefix.to_numpy(), equal_nan=True)
    signals = strategy.generate_signals(data, full, ctx, params).data
    shorter = strategy.generate_signals(data.iloc[:cut], prefix, ctx, params).data
    assert signals.iloc[:cut].equals(shorter)
    assert signals["signal"].isin([-1, 0, 1]).all() and (signals["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
