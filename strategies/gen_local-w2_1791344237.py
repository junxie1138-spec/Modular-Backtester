from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    visibility_window: int = 12
    drawdown_window: int = 40
    min_drawdown: float = 0.015
    min_visible_span: int = 4
    recovery_window: int = 3
    min_efficiency: float = 0.25
    hold_bars: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791344237"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        return int(max(params.drawdown_window, params.visibility_window + 1,
                       params.recovery_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.visibility_window < 2 or params.drawdown_window < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 2 <= params.min_visible_span <= params.visibility_window:
            raise ValueError("Visible span must fit the visibility window")
        if params.recovery_window < 1 or params.hold_bars < 1:
            raise ValueError("Recovery and holding periods must be positive")
        if not 0.0 < params.min_drawdown < 1.0:
            raise ValueError("Drawdown threshold must be between zero and one")
        if not 0.0 <= params.min_efficiency <= 1.0:
            raise ValueError("Efficiency threshold must be between zero and one")
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError("Sizes must be positive fractions no greater than one")

        close = data["close"].astype(float).where(lambda x: np.isfinite(x) & x.gt(0))
        out = pd.DataFrame(index=data.index)
        out["degree"] = 0.0
        out["visible_span"] = 0.0
        for lag in range(1, params.visibility_window + 1):
            older = close.shift(lag)
            valid = close.notna() & older.notna()
            if lag == 1:
                edge = valid
            else:
                interior = close.shift(1).rolling(lag - 1, min_periods=lag - 1).max()
                edge = valid & interior.lt(np.minimum(close, older))
            out[f"edge_{lag}"] = edge.astype(float)
            out["degree"] += out[f"edge_{lag}"]
            out["visible_span"] = np.where(edge, float(lag), out["visible_span"])

        out["span_expansion"] = out["visible_span"].diff()
        out["rolling_peak"] = close.rolling(params.drawdown_window,
                                              min_periods=params.drawdown_window).max()
        out["drawdown"] = 1.0 - close.div(out["rolling_peak"])
        out["change"] = close.diff()
        out["recovery_net"] = close.diff(params.recovery_window)
        out["recovery_path"] = out["change"].abs().rolling(
            params.recovery_window, min_periods=params.recovery_window).sum()
        out["efficiency"] = out["recovery_net"].div(
            out["recovery_path"].replace(0.0, np.nan)).clip(0.0, 1.0)
        out["topology_strength"] = (
            0.5 * out["visible_span"].div(params.visibility_window)
            + 0.5 * out["degree"].div(0.5 * params.visibility_window).clip(0.0, 1.0)
        ).clip(0.0, 1.0)
        out["conviction"] = (
            0.5 * out["topology_strength"] + 0.5 * out["efficiency"]
        ).clip(0.0, 1.0).fillna(0.0)
        out["eligible"] = (
            out["drawdown"].ge(params.min_drawdown)
            & out["visible_span"].ge(params.min_visible_span)
            & out["span_expansion"].gt(0.0)
            & out["degree"].ge(2.0)
            & out["change"].gt(0.0)
            & out["efficiency"].ge(params.min_efficiency)
        ).fillna(False)
        out.iloc[:GeneratedStrategy.warmup_bars(params),
                 out.columns.get_loc("eligible")] = False
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("Holding period must be positive")
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError("Invalid position sizes")
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].replace(
            [np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), params.min_size, dtype=float)
        exit_bar = -1
        entry_size = params.min_size
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                else:
                    signal[i] = 1
                    size[i] = entry_size
            elif eligible[i]:
                exit_bar = i + params.hold_bars
                entry_size = float(params.min_size + (
                    params.max_size - params.min_size) * conviction[i])
                signal[i] = 1
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=12)
    sample = pd.DataFrame({"close": np.ones(12)}, index=index)
    features = pd.DataFrame({"eligible": True,
                             "conviction": [0.0] * 6 + [1.0] * 6}, index=index)
    result = GeneratedStrategy.generate_signals(sample, features, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    assert np.allclose(result["size"].iloc[1:5], p.min_size)
    assert np.isclose(result["size"].iloc[11], p.max_size)
    graph_sample = pd.DataFrame({"close": [5.0, 1.0, 2.0, 3.0, 4.0]},
                                index=index[:5])
    graph = GeneratedStrategy.indicators(graph_sample, p)
    assert graph["edge_4"].iloc[-1] == 1.0
    assert graph["edge_2"].iloc[-1] == 0.0
    assert graph["visible_span"].iloc[-1] == 4.0
