from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 16
    regime_threshold: float = 0.60
    vote_threshold: float = 0.20
    pressure_window: int = 3
    pressure_threshold: float = 0.15
    profit_target: float = 0.025
    max_hold: int = 8

    def __post_init__(self):
        if self.graph_window < 4 or self.pressure_window < 1 or self.max_hold < 1:
            raise ValueError("Lookbacks and holding period must be positive; graph_window >= 4.")
        if not 0.5 < self.regime_threshold < 1.0:
            raise ValueError("regime_threshold must be between 0.5 and 1.")
        if not 0.0 < self.vote_threshold < 1.0 or not 0.0 < self.pressure_threshold < 1.0:
            raise ValueError("Directional thresholds must be between 0 and 1.")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790874301"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(params.graph_window + 1, params.pressure_window))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["log_return"] = np.log(data["close"].where(data["close"] > 0)).diff().replace([np.inf, -np.inf], np.nan)
        out["shock"] = out["log_return"].abs()
        out["body_pressure"] = ((data["close"] - data["open"]) / (data["high"] - data["low"]).where(data["high"] > data["low"])).replace([np.inf, -np.inf], np.nan)
        out["pressure"] = out["body_pressure"].rolling(params.pressure_window, min_periods=params.pressure_window).mean()
        out["edge_count"] = np.nan
        out["same_sign_fraction"] = np.nan
        out["graph_vote"] = np.nan
        width = params.graph_window
        if len(data) >= width:
            windows = np.lib.stride_tricks.sliding_window_view(out["log_return"].to_numpy(), width)
            heights = np.abs(windows)
            signs = np.sign(windows)
            valid = np.isfinite(windows).all(axis=1)
            counts = np.zeros(len(windows))
            same = np.zeros(len(windows))
            votes = np.zeros(len(windows))
            # Nonadjacent shocks connect when every intermediate shock is smaller than both endpoints.
            # ponytail: O(rows * window^2); use a stack graph builder if windows grow large.
            for left in range(width - 2):
                barrier = heights[:, left + 1].copy()
                for right in range(left + 2, width):
                    edge = valid & (barrier < np.minimum(heights[:, left], heights[:, right]))
                    edge &= (signs[:, left] != 0) & (signs[:, right] != 0)
                    counts += edge
                    same += edge & (signs[:, left] == signs[:, right])
                    votes += np.where(edge, signs[:, left] + signs[:, right], 0.0)
                    barrier = np.maximum(barrier, heights[:, right])
            denominator = np.where(counts > 0, counts, np.nan)
            out.loc[out.index[width - 1:], "edge_count"] = np.where(valid, counts, np.nan)
            out.loc[out.index[width - 1:], "same_sign_fraction"] = same / denominator
            out.loc[out.index[width - 1:], "graph_vote"] = votes / (2.0 * denominator)
        enough = out["edge_count"].ge(3)
        out["regime"] = np.select(
            [enough & out["same_sign_fraction"].ge(params.regime_threshold),
             enough & out["same_sign_fraction"].le(1.0 - params.regime_threshold)],
            [1, -1], default=0
        )
        out["topology_direction"] = np.where(
            out["graph_vote"].abs().ge(params.vote_threshold),
            np.sign(out["graph_vote"].fillna(0)) * out["regime"], 0
        ).astype(int)
        out["pressure_direction"] = np.select(
            [out["pressure"].ge(params.pressure_threshold),
             out["pressure"].le(-params.pressure_threshold)],
            [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        agreed = np.where(
            indicators["topology_direction"].eq(indicators["pressure_direction"]),
            indicators["topology_direction"], 0
        )
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        for bar in range(len(data)):
            if position:
                # The prior close's entry decision fills at this bar's open.
                if bar == entry_bar:
                    entry_price = opens[bar]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        raise ValueError("Entry open must be finite and positive.")
                gain = position * (closes[bar] / entry_price - 1.0)
                if gain >= params.profit_target or bar - entry_bar + 1 >= params.max_hold:
                    position = 0
                    entry_price = np.nan
                    continue
            elif bar >= self.warmup_bars(params) - 1 and agreed[bar] != 0:
                position = int(agreed[bar])
                entry_bar = bar + 1
            decisions[bar] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params(graph_window=4, pressure_window=1, max_hold=2, profit_target=0.02)
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 98.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"topology_direction": 0, "pressure_direction": 0}, index=index)
    indicators.iloc[4] = 1
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    data.loc[index[5], "close"] = 103.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[5] == 1 and result["signal"].iloc[6] == 0
    indicators.iloc[4] = -1
    data.loc[index[5], "close"] = 97.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[5] == -1 and result["signal"].iloc[6] == 0
    indicators.iloc[4] = [1, -1]
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].eq(0).all()
    calculated = strategy.indicators(data, params)
    assert calculated.index.equals(data.index)
    assert calculated["topology_direction"].isin([-1, 0, 1]).all()


if __name__ == "__main__":
    _self_check()
