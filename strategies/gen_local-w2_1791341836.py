from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_window: int = 20
    pulse_window: int = 5
    baseline_window: int = 20
    expansion_multiple: float = 1.1
    diffusion_gap: float = 0.25
    close_threshold: float = 0.65
    atr_window: int = 14
    stop_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791341836"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.range_window + params.pulse_window + params.baseline_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.range_window, params.pulse_window, params.baseline_window, params.atr_window, params.holding_bars) < 1:
            raise ValueError("Window lengths and holding_bars must be positive")
        if not (params.expansion_multiple > 0 and params.stop_atr > 0 and 0 < params.diffusion_gap <= 1 and 0.5 < params.close_threshold < 1):
            raise ValueError("Invalid expansion, diffusion, close-location, or stop parameter")
        out = pd.DataFrame(index=data.index)
        out["range"] = data["high"] - data["low"]
        out["range_baseline"] = out["range"].rolling(params.range_window).median().shift(1)
        out["close_location"] = (data["close"] - data["low"]) / out["range"].where(out["range"] > 0)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([out["range"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        valid = out["range_baseline"].gt(0) & out["close_location"].notna() & previous_close.notna()
        expanded = out["range"] > params.expansion_multiple * out["range_baseline"]
        out["up_event"] = (expanded & out["close_location"].ge(params.close_threshold) & data["close"].gt(previous_close)).astype(float).where(valid)
        out["down_event"] = (expanded & out["close_location"].le(1.0 - params.close_threshold) & data["close"].lt(previous_close)).astype(float).where(valid)
        for direction in ("up", "down"):
            events = out[direction + "_event"]
            out[direction + "_pulse"] = events.rolling(params.pulse_window).mean()
            out[direction + "_baseline"] = events.shift(params.pulse_window).rolling(params.baseline_window).mean()
            out[direction + "_candidate"] = (out[direction + "_pulse"].ge(0.6) & (out[direction + "_pulse"] - out[direction + "_baseline"]).ge(params.diffusion_gap)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        up = indicators["up_candidate"].fillna(False).astype(bool)
        down = indicators["down_candidate"].fillna(False).astype(bool)
        confirmed_up = up & up.shift(1, fill_value=False)
        confirmed_down = down & down.shift(1, fill_value=False)
        candidates = np.select([confirmed_up & ~confirmed_down, confirmed_down & ~confirmed_up], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        fixed_stop = np.nan
        for i in range(len(data)):
            # A decision from the previous close fills at this bar's open.
            if position == 0 and i > 0 and decisions[i - 1] != 0:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    position = int(decisions[i - 1])
                    entry_bar = i
                    fixed_stop = opens[i] - position * params.stop_atr * atr[i - 1]
            if position != 0:
                stopped = np.isfinite(closes[i]) and ((position == 1 and closes[i] < fixed_stop) or (position == -1 and closes[i] > fixed_stop))
                expired = i - entry_bar + 1 >= params.holding_bars
                if stopped or expired:
                    position = 0
                    continue
                decisions[i] = position
            elif candidates[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                decisions[i] = int(candidates[i])
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    closes = np.array([100, 100, 101, 96, 100, 100, 99, 104], dtype=float)
    opens = np.array([100, 100, 102, 101, 100, 100, 98, 99], dtype=float)
    data = pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + 1, "low": np.minimum(opens, closes) - 1, "close": closes, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 2.0, "up_candidate": [True, True, False, False, False, False, False, False], "down_candidate": [False, False, False, False, True, True, False, False]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, Params()).data
    # Checks confirmation, one-bar delay, actual opening entry prices, and both stops.
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, -1, -1]
    assert result["size"].gt(0).all()
    expired = strategy.generate_signals(data, indicators, None, Params(holding_bars=1)).data
    assert expired["signal"].tolist() == [0, 0, 1, 0, 0, 0, -1, 0]


if __name__ == "__main__":
    _self_check()
