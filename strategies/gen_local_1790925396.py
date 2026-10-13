from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 18
    peak_window: int = 5
    mean_window: int = 40
    minimum_peak: float = 2.5
    fade_threshold: float = 0.20
    minimum_drawdown: float = 0.015
    profit_target: float = 0.04
    time_stop: int = 18


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790925396"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        if min(params.graph_window, params.peak_window, params.mean_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 < params.fade_threshold < 1 and 0 < params.minimum_drawdown < 1):
            raise ValueError("Fade and drawdown thresholds must lie between zero and one.")
        if params.minimum_peak <= 0 or params.profit_target <= 0 or params.time_stop < 1:
            raise ValueError("Peak, target and holding period must be positive.")
        return int(max(params.mean_window, params.graph_window + params.peak_window + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"].astype(float).where(data["close"] > 0)
        log_return = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        loss = (-log_return).clip(lower=0)
        radius = pd.Series(np.nan, index=data.index, dtype=float)
        width = params.graph_window
        if len(data) >= width:
            windows = np.lib.stride_tricks.sliding_window_view(loss.to_numpy(), width)
            valid = np.isfinite(windows).all(axis=1)
            values = np.nan_to_num(windows, nan=0.0, posinf=0.0, neginf=0.0)
            # Horizontal visibility, then retain only edges joining negative-return nodes.
            adjacency = np.zeros((len(windows), width, width), dtype=float)
            for left in range(width - 1):
                barrier = np.full(len(windows), -np.inf)
                for right in range(left + 1, width):
                    visible = (
                        (np.minimum(values[:, left], values[:, right]) > barrier)
                        & (values[:, left] > 0)
                        & (values[:, right] > 0)
                    )
                    adjacency[:, left, right] = visible
                    adjacency[:, right, left] = visible
                    barrier = np.maximum(barrier, values[:, right])
            eigenvalues = np.linalg.eigvalsh(adjacency)
            spectral_radius = eigenvalues[:, -1].clip(min=0.0)
            spectral_radius[~valid] = np.nan
            radius.iloc[width - 1:] = spectral_radius
        prior_peak = radius.shift(1).rolling(params.peak_window, min_periods=params.peak_window).max()
        fade = (1.0 - radius / prior_peak.where(prior_peak > 0)).clip(0.0, 1.0)
        mean = close.rolling(params.mean_window, min_periods=params.mean_window).mean()
        drawdown = (1.0 - close / mean).clip(lower=0.0)
        eligible = (
            (prior_peak >= params.minimum_peak)
            & (fade >= params.fade_threshold)
            & (radius < radius.shift(1))
            & (drawdown >= params.minimum_drawdown)
        ).fillna(False)
        strength = (
            0.25
            + 0.70
            * (fade / 0.50).clip(0.0, 1.0)
            * (drawdown / (2.0 * params.minimum_drawdown)).clip(0.0, 1.0)
        ).fillna(0.25).clip(0.25, 0.95)
        return pd.DataFrame(
            {
                "log_return": log_return,
                "loss": loss,
                "spectral_radius": radius,
                "prior_peak": prior_peak,
                "fade": fade,
                "mean": mean,
                "drawdown": drawdown,
                "eligible": eligible,
                "strength": strength,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.25, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].fillna(0.25).clip(0.25, 0.95).to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        held_size = 0.25
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # The order decided at the prior close fills at today's open.
                    entry_price = opens[i]
                expired = i - entry_bar + 1 >= params.time_stop
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if expired or invalid_entry or target_hit:
                    active = False
            elif eligible[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                held_size = strength[i]
            if active:
                signal[i] = 1
                size[i] = held_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 106.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    derived = pd.DataFrame(
        {"eligible": [True] + [False] * 7, "strength": 0.80}, index=index
    )
    strategy = GeneratedStrategy()
    params = VisibilityParams(time_stop=3)
    timed = strategy.generate_signals(data, derived, None, params).data
    assert timed["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    assert np.allclose(timed.loc[timed["signal"] == 1, "size"], 0.80)
    data.loc[index[2], "close"] = 105.0
    targeted = strategy.generate_signals(data, derived, None, params).data
    assert targeted["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    # A gap at the actual entry must reset the target's price reference.
    data.loc[index[1], "open"] = 110.0
    gapped = strategy.generate_signals(data, derived, None, params).data
    assert gapped["signal"].tolist() == timed["signal"].tolist()
    empty = data.iloc[:0]
    result = strategy.generate_signals(empty, strategy.indicators(empty, params), None, params)
    assert result.data.empty


if __name__ == "__main__":
    _self_check()
