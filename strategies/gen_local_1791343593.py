from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    energy_window: int = 5
    baseline_window: int = 60
    range_window: int = 10
    compression_threshold: float = 0.55
    profit_target: float = 0.025
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791343593"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        if min(params.energy_window, params.baseline_window, params.range_window, params.holding_bars) < 1:
            raise ValueError("Windows and holding_bars must be positive")
        if not 0.0 < params.compression_threshold < 1.0:
            raise ValueError("compression_threshold must lie between zero and one")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive")
        return int(8 + params.energy_window + params.baseline_window + params.range_window + 6)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        ratio_columns = []
        for scale in (2, 4, 8):
            half = scale // 2
            recent = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{scale}"] = (recent - recent.shift(half)) * np.sqrt(scale) / 2.0
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
            out[f"baseline_{scale}"] = out[f"energy_{scale}"].shift(1).rolling(
                params.baseline_window, min_periods=params.baseline_window
            ).mean()
            name = f"ratio_{scale}"
            out[name] = out[f"energy_{scale}"] / out[f"baseline_{scale}"].where(
                out[f"baseline_{scale}"] > 0.0
            )
            ratio_columns.append(name)
        out["maximum_energy_ratio"] = out[ratio_columns].max(axis=1, skipna=False)
        out["range_high"] = data["high"].rolling(
            params.range_window, min_periods=params.range_window
        ).max()
        out["range_low"] = data["low"].rolling(
            params.range_window, min_periods=params.range_window
        ).min()
        out["range_width"] = (out["range_high"] - out["range_low"]) / data["close"].where(
            data["close"] > 0.0
        )
        out["baseline_width"] = out["range_width"].shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        out["compressed"] = (
            out["maximum_energy_ratio"].lt(params.compression_threshold)
            & out["range_width"].lt(out["baseline_width"])
        ).fillna(False)
        out["armed"] = out["compressed"].astype(float).shift(2).rolling(
            4, min_periods=4
        ).max().eq(1.0)
        out["prior_high"] = out["range_high"].shift(1)
        out["prior_low"] = out["range_low"].shift(1)
        out["up_escape"] = data["close"].gt(out["prior_high"]).fillna(False)
        out["down_escape"] = data["close"].lt(out["prior_low"]).fillna(False)
        out["long_confirmed"] = (
            out["armed"] & out["up_escape"] & out["up_escape"].shift(1, fill_value=False)
        )
        out["short_confirmed"] = (
            out["armed"] & out["down_escape"] & out["down_escape"].shift(1, fill_value=False)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_confirmed"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_confirmed"].fillna(False).to_numpy(dtype=bool)
        position = 0
        entry_price = np.nan
        held = 0
        pending_entry = False
        for i in range(len(data)):
            if pending_entry:
                entry_price = opens[i]
                pending_entry = False
                held = 0
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    position = 0
                    continue
            if position != 0:
                held += 1
                gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or held >= params.holding_bars:
                    position = 0
                    entry_price = np.nan
                    held = 0
                signals[i] = position
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                pending_entry = True
                signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=6)
    for direction in (1, -1):
        close = np.array([100.0, 100.0, 100.0 + direction, 100.0 + 3.0 * direction, 100.0, 100.0])
        data = pd.DataFrame({"open": 100.0, "high": np.maximum(close, 100.0), "low": np.minimum(close, 100.0), "close": close, "volume": 1000.0}, index=index)
        confirmed = np.array([False, True, False, False, False, False])
        indicators = pd.DataFrame({"long_confirmed": confirmed if direction == 1 else False, "short_confirmed": confirmed if direction == -1 else False}, index=index)
        for target, duration in ((0.02, 8), (0.50, 2)):
            result = strategy.generate_signals(data, indicators, None, StrategyParams(profit_target=target, holding_bars=duration)).data
            assert result["signal"].tolist() == [0, 0, direction, direction, 0, 0]
            assert result["size"].gt(0.0).all()
