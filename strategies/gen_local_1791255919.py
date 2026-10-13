from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAnalogParams:
    shape_bars: int = 5
    history_bars: int = 252
    analog_horizon: int = 8
    match_quantile: float = 0.10
    entry_quantile: float = 0.70
    profit_target: float = 0.025
    hold_bars: int = 8

    def __post_init__(self):
        if min(self.shape_bars, self.history_bars, self.analog_horizon, self.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (0 < self.match_quantile <= 1 and 0 < self.entry_quantile < 1):
            raise ValueError("Invalid quantiles.")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("Profit target must be positive.")


class GeneratedStrategy(BaseStrategy[RangeAnalogParams]):
    strategy_id = "gen_local_1791255919"

    @classmethod
    def params_type(cls):
        return RangeAnalogParams

    @staticmethod
    def warmup_bars(params: RangeAnalogParams) -> int:
        return params.shape_bars + params.history_bars + params.analog_horizon + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeAnalogParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        out["range_fraction"] = (data["high"] - data["low"]).clip(lower=0) / previous_close
        out["range_mean"] = out["range_fraction"].rolling(p.shape_bars, min_periods=p.shape_bars).mean()
        columns = []
        for lag in range(p.shape_bars):
            name = f"range_shape_{lag}"
            columns.append(name)
            out[name] = out["range_fraction"].shift(lag) / out["range_mean"].where(out["range_mean"] > 0)
        h = p.analog_horizon
        out["outcome_high"] = data["high"].rolling(h, min_periods=h).max()
        out["outcome_low"] = data["low"].rolling(h, min_periods=h).min()
        origin_close = data["close"].shift(h).where(lambda s: s > 0)
        out["matured_balance"] = (out["outcome_high"] + out["outcome_low"]) / origin_close - 2.0
        shapes = out[columns].to_numpy(dtype=float)
        outcomes = out["matured_balance"].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        threshold = np.full(len(data), np.nan)
        distance_cutoff = np.full(len(data), np.nan)
        match_count = np.zeros(len(data), dtype=int)
        # ponytail: bounded brute-force analog search; index shapes if histories become much larger.
        for i in range(GeneratedStrategy.warmup_bars(p) - 1, len(data)):
            if not np.isfinite(shapes[i]).all():
                continue
            # Endpoint e has an outcome at e+h; require e+h < i.
            endpoints = np.arange(i - h - p.history_bars, i - h)
            pool_shapes = shapes[endpoints]
            pool_outcomes = outcomes[endpoints + h]
            valid = np.isfinite(pool_shapes).all(axis=1) & np.isfinite(pool_outcomes)
            if valid.sum() < max(3, p.history_bars // 2):
                continue
            distances = np.mean((pool_shapes[valid] - shapes[i]) ** 2, axis=1)
            balances = pool_outcomes[valid]
            cutoff = float(np.quantile(distances, p.match_quantile))
            matches = distances <= cutoff
            if matches.sum() < 3:
                continue
            forecast[i] = float(np.median(balances[matches]))
            threshold[i] = float(np.quantile(balances, p.entry_quantile))
            distance_cutoff[i] = cutoff
            match_count[i] = int(matches.sum())
        out["forecast_balance"] = forecast
        out["entry_threshold"] = threshold
        out["distance_cutoff"] = distance_cutoff
        out["match_count"] = match_count
        out["entry"] = ((out["forecast_balance"] > out["entry_threshold"]) &
                        (out["forecast_balance"] > 0)).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeAnalogParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            # A prior close's entry instruction fills at today's open.
            if entry_bar < 0 and i > 0 and raw[i - 1] == 1:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                entry_bar, entry_price = i, opens[i]
            if entry_bar >= 0:
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                expired = i - entry_bar + 1 >= params.hold_bars
                if target_hit or expired:
                    entry_bar, entry_price = -1, np.nan
                    raw[i] = 0
                else:
                    raw[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    p = RangeAnalogParams(hold_bars=3)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": [True, False, False, False, False, False]}, index=index)
    timed = GeneratedStrategy.generate_signals(data, entries, None, p).data
    assert timed["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 103.0
    targeted = GeneratedStrategy.generate_signals(data, entries, None, p).data
    assert targeted["signal"].tolist() == [0, 1, 0, 0, 0, 0]
    assert (targeted["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
