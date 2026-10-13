from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    edge_threshold: float = 0.002
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791328697"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 264

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0))
        for width in (2, 4, 8):
            half = width // 2
            recent = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{width}"] = (recent - recent.shift(half)) * np.sqrt(width) / 2.0
            out[f"energy_{width}"] = out[f"haar_{width}"].pow(2)
        energy = out[["energy_2", "energy_4", "energy_8"]]
        out["total_energy"] = energy.sum(axis=1, min_count=3)
        valid_energy = np.isfinite(energy).all(axis=1) & out["total_energy"].gt(0)
        out["scale"] = energy.fillna(-1.0).to_numpy().argmax(axis=1)
        out["scale"] = out["scale"].where(valid_energy)
        out["weekday"] = data.index.dayofweek.astype(float)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        # At t, this outcome belongs to the decision made at t-4.
        out["matured_return"] = (close / close.shift(4) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["origin_weekday"] = out["weekday"].shift(4)
        out["origin_scale"] = out["scale"].shift(4)
        out["forecast"] = np.nan
        for weekday in range(5):
            for scale in range(3):
                label = f"w{weekday}_s{scale}"
                matched = (
                    out["origin_weekday"].eq(weekday)
                    & out["origin_scale"].eq(scale)
                    & out["matured_return"].notna()
                )
                out[f"count_{label}"] = matched.astype(float).rolling(252, min_periods=252).sum()
                out[f"sum_{label}"] = out["matured_return"].where(matched, 0.0).rolling(252, min_periods=252).sum()
                count = out[f"count_{label}"]
                out[f"mean_{label}"] = out[f"sum_{label}"] / count.where(count >= 8)
                current = out["weekday"].eq(weekday) & out["scale"].eq(scale)
                out["forecast"] = out["forecast"].where(~current, out[f"mean_{label}"])
        out["forecast"] = out["forecast"].replace([np.inf, -np.inf], np.nan)
        out.loc[np.arange(len(out)) < self.warmup_bars(params), "forecast"] = np.nan
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if not np.isfinite(params.edge_threshold) or params.edge_threshold <= 0:
            raise ValueError("edge_threshold must be finite and positive")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry = np.nan
        entry_atr = np.nan
        fill_bar = -1
        for i in range(len(data)):
            if side != 0:
                if i == fill_bar:
                    entry = opening[i]
                valid_entry = np.isfinite(entry) and entry > 0
                stopped = (
                    valid_entry
                    and np.isfinite(close[i])
                    and side * (close[i] - entry) < -params.stop_atr * entry_atr
                )
                expired = i - fill_bar + 1 >= 4
                if not valid_entry or stopped or expired:
                    side = 0
            elif (
                np.isfinite(close[i]) and close[i] > 0
                and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(forecast[i])
                and abs(forecast[i]) >= params.edge_threshold
            ):
                side = 1 if forecast[i] > 0 else -1
                entry = np.nan
                entry_atr = atr[i]
                fill_bar = i + 1
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "forecast": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]}, index=index)
    strategy = GeneratedStrategy()
    params = Params()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    data.loc[index[2], "close"] = 98.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    indicators.loc[index[0], "forecast"] = -0.01
    data.loc[index[2], "close"] = 102.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
