from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapFlowParams:
    atr_window: int = 14
    flow_window: int = 10
    gap_threshold: float = 0.6
    flow_threshold: float = 0.35
    stop_atr: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapFlowParams]):
    strategy_id = "gen_local_1791338964"

    @classmethod
    def params_type(cls) -> type[GapFlowParams]:
        return GapFlowParams

    def warmup_bars(self, params: GapFlowParams) -> int:
        return int(max(params.atr_window, params.flow_window) + 1)

    def indicators(self, data: pd.DataFrame, params: GapFlowParams) -> pd.DataFrame:
        if params.atr_window < 1 or params.flow_window < 1:
            raise ValueError("Indicator windows must be positive")
        if not (np.isfinite(params.gap_threshold) and params.gap_threshold > 0):
            raise ValueError("gap_threshold must be finite and positive")
        if not (np.isfinite(params.flow_threshold) and 0 < params.flow_threshold <= 1):
            raise ValueError("flow_threshold must be in (0, 1]")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["intraday_return"] = data["close"].div(data["open"].where(data["open"] > 0)) - 1.0
        out["flow_net"] = out["intraday_return"].rolling(params.flow_window, min_periods=params.flow_window).sum().shift(1)
        out["flow_gross"] = out["intraday_return"].abs().rolling(params.flow_window, min_periods=params.flow_window).sum().shift(1)
        out["flow_efficiency"] = out["flow_net"].div(out["flow_gross"].where(out["flow_gross"] > 0))
        out["gap_atr"] = (data["open"] - out["prior_close"]).div(out["prior_atr"].where(out["prior_atr"] > 0))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapFlowParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")
        if not (np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("stop_atr must be finite and positive")
        long_entry = (indicators["gap_atr"] <= -params.gap_threshold) & (indicators["flow_efficiency"] >= params.flow_threshold)
        short_entry = (indicators["gap_atr"] >= params.gap_threshold) & (indicators["flow_efficiency"] <= -params.flow_threshold)
        valid = indicators["atr"].gt(0) & np.isfinite(indicators["atr"]) & data["close"].gt(0) & np.isfinite(data["close"])
        candidates = np.select([long_entry & valid, short_entry & valid], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        direction = 0
        entry_price = 0.0
        stop_distance = 0.0
        entry_bar = 0
        for i in range(len(data)):
            # A previous close's decision becomes a position at this bar's open.
            if direction == 0 and i > 0 and decisions[i - 1] != 0:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    direction = int(decisions[i - 1])
                    entry_price = opens[i]
                    stop_distance = params.stop_atr * atr[i - 1]
                    entry_bar = i
                else:
                    continue
            if direction != 0:
                # Freeze both entry price and ATR; shorts use the symmetric stop.
                stopped = not np.isfinite(closes[i]) or direction * (closes[i] - entry_price) < -stop_distance
                expired = i - entry_bar + 1 >= params.holding_bars
                if stopped or expired:
                    direction = 0
                    decisions[i] = 0
                else:
                    decisions[i] = direction
            else:
                decisions[i] = int(candidates[i])
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=22, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 111.0, "low": 89.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[[2, 3, 4, 7, 8, 9]], "close"] = [110.0, 98.5, 97.0, 90.0, 101.5, 103.0]
    ind = pd.DataFrame({"atr": 1.0, "gap_atr": np.nan, "flow_efficiency": np.nan}, index=index)
    ind.loc[index[[1, 6, 11]], "gap_atr"] = [-0.8, 0.8, -0.8]
    ind.loc[index[[1, 6, 11]], "flow_efficiency"] = [0.6, -0.6, 0.6]
    strategy = GeneratedStrategy()
    params = GapFlowParams()
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, -1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    ind.loc[index[1], "flow_efficiency"] = -0.6
    assert strategy.generate_signals(data, ind, None, params).data["signal"].iloc[2] == 0
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived["gap_atr"].iloc[:params.atr_window].isna().all()
    assert strategy.warmup_bars(params) >= max(params.atr_window, params.flow_window) + 1


if __name__ == "__main__":
    _self_check()
