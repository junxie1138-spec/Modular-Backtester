from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapShapeParams:
    shape_bars: int = 10
    calibration_bars: int = 126
    history_bars: int = 252
    hold_bars: int = 18
    atr_bars: int = 14
    min_analogs: int = 8
    confidence_quantile: float = 0.70
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[GapShapeParams]):
    strategy_id = "gen_local-w2_1791344407"

    @classmethod
    def params_type(cls) -> type[GapShapeParams]:
        return GapShapeParams

    def warmup_bars(self, params: GapShapeParams) -> int:
        return int(max(params.atr_bars + 1, params.shape_bars + 2 * params.calibration_bars + params.history_bars + params.hold_bars + 3))

    def indicators(self, data: pd.DataFrame, params: GapShapeParams) -> pd.DataFrame:
        p = params
        if (p.shape_bars < 2 or p.calibration_bars < 3 or p.history_bars < p.min_analogs
                or p.min_analogs < 1 or p.hold_bars < 1 or p.atr_bars < 1
                or not 0 < p.confidence_quantile < 1 or not np.isfinite(p.stop_atr)
                or p.stop_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        prior = data["close"].shift(1).replace(0, np.nan)
        out["gap"] = (data["open"] / prior - 1).replace([np.inf, -np.inf], np.nan)
        out["gap_mass"] = out["gap"].abs().rolling(p.shape_bars).sum()
        out["gap_total"] = out["gap"].rolling(p.shape_bars).sum()
        out["gap_recent"] = out["gap"].rolling(p.shape_bars // 2).sum()
        scale = out["gap_mass"].replace(0, np.nan)
        out["balance"] = out["gap_total"] / scale
        out["bend"] = (out["gap_recent"] * p.shape_bars / (p.shape_bars // 2) - out["gap_total"]) / scale
        valid = pd.Series(True, index=data.index)
        codes = []
        for name in ("balance", "bend"):
            past = out[name].shift(1).rolling(p.calibration_bars)
            out[name + "_low"] = past.quantile(1 / 3)
            out[name + "_high"] = past.quantile(2 / 3)
            valid &= out[[name, name + "_low", name + "_high"]].notna().all(axis=1)
            codes.append((out[name] > out[name + "_low"]).astype(int) + (out[name] > out[name + "_high"]).astype(int))
        out["cohort"] = (3 * codes[0] + codes[1]).where(valid)
        # Only outcomes whose entire holding interval has finished enter the cohort.
        out["mature_return"] = (data["close"] / data["open"].shift(p.hold_bars - 1).replace(0, np.nan) - 1).replace([np.inf, -np.inf], np.nan)
        out["mature_cohort"] = out["cohort"].shift(p.hold_bars)
        out["forecast"] = np.nan
        for cohort in range(9):
            eligible = out["mature_cohort"].eq(cohort) & out["mature_return"].notna()
            out[f"count_{cohort}"] = eligible.astype(float).rolling(p.history_bars, min_periods=1).sum()
            out[f"mean_{cohort}"] = out["mature_return"].where(eligible, 0.0).rolling(p.history_bars, min_periods=1).sum() / out[f"count_{cohort}"].replace(0, np.nan)
            match = out["cohort"].eq(cohort) & out[f"count_{cohort}"].ge(p.min_analogs)
            out.loc[match, "forecast"] = out.loc[match, f"mean_{cohort}"]
        out["threshold"] = out["forecast"].abs().shift(1).rolling(p.calibration_bars).quantile(p.confidence_quantile)
        out["true_range"] = pd.concat([data["high"] - data["low"], (data["high"] - prior).abs(), (data["low"] - prior).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapShapeParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_atr = np.nan
        stop = np.nan
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        threshold = indicators["threshold"].to_numpy(dtype=float)
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                    else:
                        stop = opens[i] - position * params.stop_atr * entry_atr
                if position and (not np.isfinite(closes[i]) or position * (closes[i] - stop) < 0 or i - entry_bar + 1 >= params.hold_bars):
                    position = 0
            elif (np.isfinite(forecast[i]) and np.isfinite(threshold[i])
                    and np.isfinite(atr[i]) and atr[i] > 0
                    and abs(forecast[i]) > 0 and abs(forecast[i]) >= threshold[i]
                    and np.isfinite(closes[i]) and closes[i] > 0):
                position = 1 if forecast[i] > 0 else -1
                entry_bar = i + 1
                entry_atr = atr[i]
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapShapeParams()
    index = pd.date_range("2020-01-01", periods=6)
    for direction in (1, -1):
        data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 94.0, "close": [100.0, 100.0, 100.0 - direction * 5, 100.0, 100.0, 100.0], "volume": 1000.0}, index=index)
        indicators = pd.DataFrame({"atr": 1.0, "forecast": [float(direction), 0.0, 0.0, 0.0, 0.0, 0.0], "threshold": 0.5}, index=index)
        signals = strategy.generate_signals(data, indicators, None, params).data
        assert signals["signal"].tolist() == [0, direction, direction, 0, 0, 0]
        assert signals["size"].gt(0).all()
