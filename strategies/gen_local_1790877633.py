from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEchoParams:
    entry_strength: float = 0.75
    profit_target: float = 0.015


class GeneratedStrategy(BaseStrategy[GapEchoParams]):
    strategy_id = "gen_local_1790877633"

    @classmethod
    def params_type(cls) -> type[GapEchoParams]:
        return GapEchoParams

    def warmup_bars(self, params: GapEchoParams) -> int:
        return 63

    def indicators(self, data: pd.DataFrame, params: GapEchoParams) -> pd.DataFrame:
        if not np.isfinite(params.entry_strength) or params.entry_strength <= 0:
            raise ValueError("entry_strength must be finite and positive")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        out = pd.DataFrame(index=data.index)
        prior = data["close"].shift(1)
        out["gap"] = (data["open"] / prior.where(prior > 0) - 1).replace([np.inf, -np.inf], np.nan)
        out["lag1"] = out["gap"].shift(1)
        out["lag2"] = out["gap"].shift(2)
        valid = out[["gap", "lag1", "lag2"]].notna().all(axis=1)
        y = out["gap"].where(valid)
        x = out["lag1"].where(valid)
        z = out["lag2"].where(valid)
        for name, series in (("mean_gap", y), ("mean_lag1", x), ("mean_lag2", z)):
            out[name] = series.rolling(60, min_periods=60).mean()
        out["var1"] = x.rolling(60, min_periods=60).var()
        out["var2"] = z.rolling(60, min_periods=60).var()
        out["cov12"] = x.rolling(60, min_periods=60).cov(z)
        out["cov_y1"] = y.rolling(60, min_periods=60).cov(x)
        out["cov_y2"] = y.rolling(60, min_periods=60).cov(z)
        out["determinant"] = out["var1"] * out["var2"] - out["cov12"] ** 2
        denominator = out["determinant"].where(out["determinant"] > 1e-20)
        out["a"] = (out["cov_y1"] * out["var2"] - out["cov_y2"] * out["cov12"]) / denominator
        out["b"] = (out["cov_y2"] * out["var1"] - out["cov_y1"] * out["cov12"]) / denominator
        out["intercept"] = out["mean_gap"] - out["a"] * out["mean_lag1"] - out["b"] * out["mean_lag2"]
        out["gap_std"] = y.rolling(60, min_periods=60).std()
        previous, current = out["lag1"], out["gap"]
        for step in range(1, 6):
            out[f"forecast_{step}"] = out["intercept"] + out["a"] * current + out["b"] * previous
            previous, current = current, out[f"forecast_{step}"]
        out["forecast_sum"] = out[[f"forecast_{step}" for step in range(2, 6)]].sum(axis=1, min_count=4)
        out["strength"] = out["forecast_sum"] / (2 * out["gap_std"].where(out["gap_std"] > 1e-8))
        out["eligible"] = (
            (out["b"].abs() > out["a"].abs())
            & (out["b"].abs() < 1)
            & (out["a"] + out["b"] < 1)
            & (out["b"] - out["a"] < 1)
            & np.isfinite(out["strength"])
        )
        out["entry"] = np.where(
            out["eligible"] & (out["strength"].abs() >= params.entry_strength),
            np.sign(out["strength"].fillna(0)),
            0,
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapEchoParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        side = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    entry_price = opens[i]
                gain = side * (closes[i] / entry_price - 1) if entry_price > 0 else np.nan
                if gain >= params.profit_target or i - entry_bar + 1 >= 4:
                    side = 0
                    entry_price = np.nan
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(entries[i])
                entry_bar = i + 1
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapEchoParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [1, 0, 0, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    for side in (1, -1):
        indicators.iloc[0, 0] = side
        data.loc[index[1], "close"] = 100 * (1 + side * 0.02)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, side, 0, 0, 0, 0, 0, 0]
    flat = data.copy()
    flat["close"] = 100.0
    assert strategy.indicators(flat, params)["entry"].eq(0).all()


if __name__ == "__main__":
    self_check()
