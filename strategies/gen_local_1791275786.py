from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    atr_window: int = 14
    regime_window: int = 63
    shape_bars: int = 6
    history_bars: int = 252
    neighbors: int = 12
    edge_atr: float = 0.12
    max_distance: float = 0.65
    stop_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local_1791275786"

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(params.atr_window + params.regime_window +
                   params.shape_bars + params.history_bars + 3)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        if (min(params.atr_window, params.regime_window, params.shape_bars,
                params.history_bars, params.neighbors) < 1
                or params.history_bars <= params.shape_bars + 3
                or not np.isfinite([params.edge_atr, params.max_distance,
                                    params.stop_atr]).all()
                or params.max_distance <= 0 or params.stop_atr <= 0):
            raise ValueError("Invalid shape-analog parameters")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        atr = out["atr"].where(out["atr"] > 0)
        close = data["close"].where(data["close"] > 0)
        out["atr_fraction"] = atr / close
        out["vol_baseline"] = out["atr_fraction"].rolling(
            params.regime_window, min_periods=params.regime_window).median()
        out["vol_regime"] = np.log(
            out["atr_fraction"] / out["vol_baseline"].where(out["vol_baseline"] > 0))
        out["range_pulse"] = np.log(
            (out["true_range"] / atr).where(out["true_range"] > 0))
        columns = []
        for lag in range(params.shape_bars):
            name = f"pulse_{lag}"
            out[name] = out["range_pulse"].shift(lag)
            columns.append(name)
        columns.append("vol_regime")
        # At t, this label belongs to the shape at t-3; both opens are observed.
        out["matured_outcome"] = (
            data["open"] / data["open"].shift(2).where(data["open"].shift(2) > 0) - 1
        ) * close.shift(3) / atr.shift(3)
        features = out[columns].to_numpy(dtype=float)
        outcomes = out["matured_outcome"].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        match_distance = np.full(len(data), np.nan)
        # ponytail: bounded O(n * history * shape); index neighbors if histories grow.
        for i in range(self.warmup_bars(params) - 1, len(data)):
            if not np.isfinite(features[i]).all():
                continue
            # Separate shapes and complete their two-session outcomes before matching.
            candidates = np.arange(max(0, i - params.history_bars),
                                   i - params.shape_bars - 2)
            valid = (np.isfinite(features[candidates]).all(axis=1)
                     & np.isfinite(outcomes[candidates + 3]))
            candidates = candidates[valid]
            if len(candidates) < params.neighbors:
                continue
            distance = np.sqrt(np.mean(
                (features[candidates] - features[i]) ** 2, axis=1))
            nearest = np.argsort(distance, kind="stable")[:params.neighbors]
            match_distance[i] = float(distance[nearest].mean())
            if match_distance[i] <= params.max_distance:
                forecast[i] = float(outcomes[candidates[nearest] + 3].mean())
        out["match_distance"] = match_distance
        out["forecast_atr"] = forecast
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        qualified = (indicators["forecast_atr"] > params.edge_atr).fillna(False)
        confirmed = (qualified & qualified.shift(1, fill_value=False)).to_numpy()
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        active = False
        stop = np.nan
        held = 0
        for i in range(len(data)):
            # Yesterday's decision enters at today's open; freeze yesterday's ATR.
            if i > 0 and not active and desired[i - 1] == 1:
                if (np.isfinite(opens[i]) and opens[i] > 0
                        and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    active = True
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                    held = 0
                else:
                    continue
            if active:
                held += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or held >= 2:
                    active = False
                else:
                    desired[i] = 1
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                desired[i] = 1
        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 2.0, "forecast_atr": 1.0}, index=index)
    strategy = GeneratedStrategy()
    params = ShapeParams()
    normal = strategy.generate_signals(data, indicators, None, params).data
    assert normal["signal"].tolist()[:5] == [0, 0, 1, 1, 0]
    data.loc[index[2], "close"] = 97.0
    stopped = strategy.generate_signals(data, indicators, None, params).data
    assert stopped["signal"].tolist()[:4] == [0, 0, 1, 0]
    assert stopped["size"].gt(0).all()
    assert set(stopped["signal"].unique()) <= {0, 1}


if __name__ == "__main__":
    _self_check()
