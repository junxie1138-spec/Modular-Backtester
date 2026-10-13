from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    shape_bars: int = 4
    archive_bars: int = 252
    neighbors: int = 9
    atr_bars: int = 14
    hold_bars: int = 8
    min_gap_atr: float = 0.35
    match_radius: float = 0.75
    edge: float = 0.006
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791322164"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(200, params.atr_bars + params.shape_bars
                       + params.archive_bars + params.hold_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["ma200"] = data["close"].rolling(200).mean()
        out["gap"] = (data["open"] - previous) / out["atr"].shift(1).replace(0, np.nan)
        out["gap_sign"] = np.sign(out["gap"])
        out["regime"] = np.sign(data["close"] - out["ma200"])
        out["mirror_regime"] = out["regime"] * out["gap_sign"]
        columns = []
        for lag in range(params.shape_bars):
            column = f"gap_shape_{lag}"
            out[column] = out["gap"].shift(lag) * out["gap_sign"]
            columns.append(column)
        shapes = out[columns].to_numpy(dtype=float)
        gaps = out["gap_sign"].to_numpy(dtype=float)
        regimes = out["mirror_regime"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        finite_shapes = np.isfinite(shapes).all(axis=1)
        # ponytail: bounded archive scan; batch distances if much larger archives are needed.
        for i in range(self.warmup_bars(params) - 1, len(data)):
            if not finite_shapes[i] or not np.isfinite(regimes[i]) or gaps[i] == 0:
                continue
            end = i - params.hold_bars
            candidates = np.arange(max(0, end - params.archive_bars), end)
            valid = (finite_shapes[candidates]
                     & (regimes[candidates] == regimes[i])
                     & (gaps[candidates] != 0)
                     & np.isfinite(opening[candidates + 1])
                     & (opening[candidates + 1] > 0)
                     & np.isfinite(closing[candidates + params.hold_bars])
                     & (closing[candidates + params.hold_bars] > 0))
            candidates = candidates[valid]
            distance = np.mean((shapes[candidates] - shapes[i]) ** 2, axis=1)
            nearby = distance <= params.match_radius ** 2
            candidates, distance = candidates[nearby], distance[nearby]
            if len(candidates) < params.neighbors:
                continue
            chosen = candidates[np.argsort(distance, kind="stable")[:params.neighbors]]
            # All labels end before today's close; fills begin at the next open.
            outcomes = closing[chosen + params.hold_bars] / opening[chosen + 1] - 1.0
            forecast[i] = gaps[i] * np.mean(gaps[chosen] * outcomes)
        out["forecast"] = forecast
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        raw = np.zeros(n, dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        values = indicators[["atr", "gap", "regime", "forecast"]].to_numpy(dtype=float)
        side, pending, age = 0, 0, 0
        entry, best, stop = 0.0, 0.0, -np.inf
        armed = False
        for i in range(n):
            opening, high, low, close = prices[i]
            atr, gap, regime, forecast = values[i]
            valid_bar = (np.isfinite(prices[i]).all() and np.min(prices[i]) > 0
                         and np.isfinite(atr) and atr > 0)
            if pending:
                side, pending, age = pending, 0, 0
                entry = opening
                best, stop, armed = side * entry, -np.inf, False
            if side:
                age += 1
                # Signed price makes the protective floor ratchet upward on both sides.
                adverse = low if side == 1 else -high
                stopped = armed and adverse <= stop
                if not valid_bar or stopped or age >= params.hold_bars:
                    side = 0
                    continue
                favorable = high if side == 1 else -low
                best = max(best, favorable)
                if best - side * entry >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    # New floors become active after this close; never assume intrabar ordering.
                    stop = max(stop, side * entry, best - params.trail_atr * atr)
                    if side * close <= stop:
                        side = 0
                        continue
                raw[i] = side
                continue
            if i < self.warmup_bars(params) - 1 or not valid_bar:
                continue
            if not np.isfinite(values[i]).all() or abs(gap) < params.min_gap_atr:
                continue
            if forecast >= params.edge and regime > 0:
                pending = 1
            elif forecast <= -params.edge and regime < 0:
                pending = -1
            raw[i] = pending
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    n = strategy.warmup_bars(params) + 12
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    for side in (1, -1):
        data = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5,
                             "close": 100.0, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"atr": 1.0, "gap": 0.5, "regime": float(side),
                            "forecast": np.nan}, index=index)
        decision = strategy.warmup_bars(params) - 1
        ind.iloc[decision, ind.columns.get_loc("forecast")] = side * 0.02
        rally = decision + 1
        data.iloc[rally, data.columns.get_loc("high" if side == 1 else "low")] = 100 + side * 2
        data.iloc[rally, data.columns.get_loc("close")] = 100 + side * 1.8
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].iloc[decision] == 0
        assert result["signal"].iloc[rally] == side
        assert result["signal"].iloc[rally + 1] == side
        assert result["signal"].iloc[rally + 2] == 0
        assert result["size"].gt(0).all()
    x = np.arange(n, dtype=float)
    close = 100 + 0.03 * x + 2 * np.sin(x / 7)
    opening = close * (1 + 0.004 * np.sin(x))
    data = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) + 0.5,
                         "low": np.minimum(opening, close) - 0.5, "close": close,
                         "volume": 1000.0}, index=index)
    full = strategy.indicators(data, params)
    cut = n - 5
    prefix = strategy.indicators(data.iloc[:cut], params)
    pd.testing.assert_frame_equal(full.iloc[:cut], prefix)
    a = strategy.generate_signals(data, full, ctx, params).data
    b = strategy.generate_signals(data.iloc[:cut], prefix, ctx, params).data
    pd.testing.assert_frame_equal(a.iloc[:cut], b)
    assert a["signal"].isin([-1, 0, 1]).all()


if __name__ == "__main__":
    _self_check()
