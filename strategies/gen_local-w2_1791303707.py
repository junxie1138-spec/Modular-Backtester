from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791303707"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return 280

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["vol_fast"] = out["return"].rolling(5, min_periods=5).std()
        out["vol_slow"] = out["return"].rolling(20, min_periods=20).std()
        ratio = out["vol_fast"].div(out["vol_slow"].replace(0.0, np.nan))
        out["log_ratio"] = np.log(ratio.where(ratio > 0.0))
        out["log_level"] = np.log(out["vol_slow"].where(out["vol_slow"] > 0.0))
        columns = []
        for lag in (0, 3, 6, 9):
            column = f"shape_{lag}"
            out[column] = out["log_ratio"].shift(lag)
            columns.append(column)
        # At endpoint j+8, this label measures open[j+1] to close[j+8].
        out["matured_return"] = data["close"].div(
            data["open"].shift(7).replace(0.0, np.nan)
        ) - 1.0
        distances, outcomes = [], []
        for lag in range(20, 251, 10):
            difference = out[columns] - out[columns].shift(lag)
            distance = difference.pow(2).sum(axis=1, min_count=4) / 4.0
            distance += 0.25 * (out["log_level"] - out["log_level"].shift(lag)).pow(2)
            label = out["matured_return"].shift(lag - 8)
            distance = distance.where(np.isfinite(label))
            distances.append(distance.to_numpy())
            outcomes.append(label.to_numpy())
        distance_array = np.column_stack(distances)
        outcome_array = np.column_stack(outcomes)
        valid = np.isfinite(distance_array).all(axis=1)
        safe_distance = np.where(np.isfinite(distance_array), distance_array, np.inf)
        nearest = np.argsort(safe_distance, axis=1)[:, :7]
        selected = np.take_along_axis(outcome_array, nearest, axis=1)
        selected = np.where(np.isfinite(selected), selected, 0.0)
        out["analog_mean"] = selected.mean(axis=1)
        out["analog_std"] = selected.std(axis=1)
        out["analog_distance"] = np.take_along_axis(safe_distance, nearest, axis=1).mean(axis=1)
        out["agreement"] = np.maximum((selected > 0.0).mean(axis=1), (selected < 0.0).mean(axis=1))
        usable = (
            valid
            & (out["analog_distance"] <= 0.30)
            & (out["agreement"] >= 5.0 / 7.0)
            & (out["analog_mean"].abs() > 0.35 * out["analog_std"])
            & (out["analog_mean"].abs() >= 0.005)
        )
        out["entry"] = np.where(usable, np.sign(out["analog_mean"]), 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0 or params.max_hold < 1:
            raise ValueError("profit_target must be positive and max_hold must be at least one.")
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    entry_price = opens[i]
                gain = side * (closes[i] / entry_price - 1.0) if entry_price > 0.0 else np.nan
                if (
                    not np.isfinite(entry_price)
                    or entry_price <= 0.0
                    or gain >= params.profit_target
                    or i - entry_bar + 1 >= params.max_hold
                ):
                    side = 0
                decisions[i] = side
            elif entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0.0:
                side = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
                decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=560, freq="B")
    close = 100.0 * np.exp(np.cumsum(0.0002 + 0.009 * np.sin(np.arange(560) * 0.71)))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    params = AnalogParams()
    full = GeneratedStrategy.indicators(None, data, params)
    prefix = GeneratedStrategy.indicators(None, data.iloc[:400], params)
    pd.testing.assert_frame_equal(full.iloc[:400], prefix)
    flat = data.iloc[:14].copy()
    flat[["open", "close"]] = 100.0
    for side in (-1, 1):
        features = pd.DataFrame({"entry": side}, index=flat.index)
        result = GeneratedStrategy.generate_signals(None, flat, features, None, AnalogParams(0.02, 3)).data
        assert result["signal"].iloc[:5].tolist() == [0, side, side, side, 0]
        target_data = flat.copy()
        target_data.iloc[2, target_data.columns.get_loc("close")] = 100.0 + side * 3.0
        result = GeneratedStrategy.generate_signals(None, target_data, features, None, AnalogParams(0.02, 8)).data
        assert result["signal"].iloc[:4].tolist() == [0, side, side, 0]
        assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
