from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolumeClockParams:
    shape_bars: int = 6
    history_bars: int = 504
    neighbors: int = 12
    horizon_bars: int = 8
    volume_bars: int = 20
    percentile_bars: int = 126
    entry_percentile: float = 0.75
    atr_bars: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[VolumeClockParams]):
    strategy_id = "gen_local-w2_1791322601"

    @classmethod
    def params_type(cls):
        return VolumeClockParams

    def warmup_bars(self, params: VolumeClockParams) -> int:
        return int(max(
            params.history_bars + 2 * params.shape_bars
            + params.horizon_bars + params.percentile_bars + 1,
            params.volume_bars + params.percentile_bars + 1,
            params.atr_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: VolumeClockParams) -> pd.DataFrame:
        lengths = (params.shape_bars, params.history_bars, params.neighbors,
                   params.horizon_bars, params.volume_bars,
                   params.percentile_bars, params.atr_bars)
        if any(x < 1 for x in lengths):
            raise ValueError("Lookbacks, horizon and neighbor count must be positive.")
        if params.neighbors > params.history_bars:
            raise ValueError("neighbors cannot exceed history_bars.")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one.")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive.")
        if not np.isfinite(params.breakeven_pct) or params.breakeven_pct <= 0:
            raise ValueError("breakeven_pct must be finite and positive.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        volume = data["volume"].astype(float).where(data["volume"] >= 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return"] = returns
        prior_close = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior_close).abs(),
            (data["low"] - prior_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = tr.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        baseline = volume.shift(1).rolling(
            params.volume_bars, min_periods=params.volume_bars
        ).mean().replace(0.0, np.nan)
        out["relative_volume"] = volume / baseline
        out["volume_cutoff"] = out["relative_volume"].rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile).shift(1)

        w = params.shape_bars
        r = np.column_stack([returns.shift(k).to_numpy() for k in range(w - 1, -1, -1)])
        v = np.column_stack([volume.shift(k).to_numpy() for k in range(w - 1, -1, -1)])
        scale = np.abs(r).sum(axis=1)
        total_volume = v.sum(axis=1)
        path = np.divide(
            np.cumsum(r, axis=1), scale[:, None],
            out=np.full_like(r, np.nan), where=(scale > 0)[:, None],
        )
        clock = np.divide(
            np.cumsum(v, axis=1), total_volume[:, None],
            out=np.full_like(v, np.nan), where=(total_volume > 0)[:, None],
        )
        path = np.column_stack([np.zeros(len(data)), path])
        clock = np.column_stack([np.zeros(len(data)), clock])
        knots = np.array([0.25, 0.5, 0.75, 1.0])
        right = np.clip((clock[:, :, None] < knots).sum(axis=1), 1, w)
        left = right - 1
        c0 = np.take_along_axis(clock, left, axis=1)
        c1 = np.take_along_axis(clock, right, axis=1)
        p0 = np.take_along_axis(path, left, axis=1)
        p1 = np.take_along_axis(path, right, axis=1)
        fraction = np.divide(
            knots - c0, c1 - c0, out=np.full_like(c0, np.nan), where=c1 > c0
        )
        shapes = p0 + fraction * (p1 - p0)
        shapes[~(np.isfinite(r).all(axis=1) & np.isfinite(v).all(axis=1))] = np.nan
        for k, name in enumerate(("shape_q1", "shape_q2", "shape_q3", "shape_end")):
            out[name] = shapes[:, k]

        realized = close.pct_change(params.horizon_bars, fill_method=None).to_numpy()
        forecasts = np.full(len(data), np.nan)
        valid_shape = np.isfinite(shapes).all(axis=1)
        first = params.history_bars + 2 * w + params.horizon_bars - 1
        # ponytail: bounded exhaustive neighbor scan; use an index if histories become large.
        for i in range(first, len(data)):
            if not valid_shape[i]:
                continue
            end = i - w - params.horizon_bars
            candidates = np.arange(end - params.history_bars + 1, end + 1)
            outcomes = realized[candidates + params.horizon_bars]
            valid = valid_shape[candidates] & np.isfinite(outcomes)
            candidates, outcomes = candidates[valid], outcomes[valid]
            if len(candidates) < params.neighbors:
                continue
            distance = np.square(shapes[candidates] - shapes[i]).sum(axis=1)
            nearest = np.argsort(distance, kind="stable")[:params.neighbors]
            forecasts[i] = float(np.median(outcomes[nearest]))
        out["forecast"] = forecasts
        out["forecast_cutoff"] = out["forecast"].rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: VolumeClockParams,
    ) -> SignalFrame:
        eligible = (
            (indicators["forecast"] > indicators["forecast_cutoff"])
            & (indicators["forecast"] > 0)
            & (indicators["relative_volume"] > indicators["volume_cutoff"])
            & (indicators["return"] > 0)
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = peak = stop = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if not (np.isfinite(opens[i]) and opens[i] > 0
                        and i > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                entry = peak = opens[i]
                stop = entry - params.trail_atr * atr[i - 1]
                active, armed, age = True, False, 0
            if active:
                age += 1
                price = closes[i]
                if np.isfinite(price) and price > 0:
                    peak = max(peak, price)
                    armed = armed or (np.isfinite(highs[i]) and
                                      highs[i] >= entry * (1.0 + params.breakeven_pct))
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, peak - params.trail_atr * atr[i])
                    # Stops are evaluated at closes; exits execute at the next open.
                    if price <= stop:
                        active = False
                if age >= params.horizon_bars:
                    active = False
                signal[i] = int(active)
                continue
            if eligible[i]:
                pending = True
                signal[i] = 1

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    close = np.array([100.0, 100.0, 103.0, 102.0, 100.5, 100.0, 100.0])
    data = pd.DataFrame({
        "open": 100.0, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "forecast": [0.1] + [np.nan] * 6, "forecast_cutoff": 0.05,
        "relative_volume": 2.0, "volume_cutoff": 1.0,
        "return": 0.01, "atr": [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = GeneratedStrategy().generate_signals(data, ind, ctx, VolumeClockParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
