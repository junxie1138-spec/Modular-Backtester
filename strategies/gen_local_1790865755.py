from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalDebtParams:
    season_window: int = 504
    debt_window: int = 3
    min_samples: int = 40
    edge_ratio: float = 0.08
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[SeasonalDebtParams]):
    strategy_id = "gen_local_1790865755"

    @classmethod
    def params_type(cls) -> type[SeasonalDebtParams]:
        return SeasonalDebtParams

    def warmup_bars(self, params: SeasonalDebtParams) -> int:
        return int(max(200, params.season_window + 1,
                       params.debt_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: SeasonalDebtParams) -> pd.DataFrame:
        if (params.season_window < 2 or params.debt_window < 1
                or params.atr_window < 1 or params.min_samples < 2
                or params.min_samples > params.season_window
                or params.hold_bars < 1 or not np.isfinite(params.stop_atr)
                or params.stop_atr <= 0 or not np.isfinite(params.edge_ratio)
                or params.edge_ratio < 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["phase"] = np.minimum((data.index.day - 1) // 7, 3)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["volatility"] = out["return"].rolling(
            params.season_window, min_periods=params.season_window
        ).std()
        out["season_mean"] = np.nan
        out["season_count"] = 0.0
        for phase in range(4):
            mask = out["phase"].eq(phase)
            selected = out["return"].where(mask)
            count = selected.notna().astype(float).rolling(
                params.season_window, min_periods=params.season_window
            ).sum()
            mean = selected.fillna(0.0).rolling(
                params.season_window, min_periods=params.season_window
            ).sum() / count.where(count >= params.min_samples)
            out[f"phase_{phase}_count"] = count
            out[f"phase_{phase}_mean"] = mean
            out.loc[mask, "season_mean"] = mean.loc[mask]
            out.loc[mask, "season_count"] = count.loc[mask]
        out["recent_return"] = out["return"].rolling(
            params.debt_window, min_periods=params.debt_window
        ).sum()
        out["debt"] = out["season_mean"] * params.debt_window - out["recent_return"]
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SeasonalDebtParams
    ) -> SignalFrame:
        ready = np.isfinite(indicators[
            ["season_mean", "volatility", "debt", "ma200", "atr"]
        ]).all(axis=1) & indicators["atr"].gt(0) & indicators["volatility"].gt(0)
        threshold = params.edge_ratio * indicators["volatility"]
        long_entry = (ready & data["close"].gt(indicators["ma200"])
                      & indicators["season_mean"].gt(threshold)
                      & indicators["debt"].gt(0))
        short_entry = (ready & data["close"].lt(indicators["ma200"])
                       & indicators["season_mean"].lt(-threshold)
                       & indicators["debt"].lt(0))
        candidate = np.select([long_entry, short_entry], [1, -1], default=0)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        side, entry_bar = 0, -1
        entry, stop = np.nan, np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    entry = opening[i]
                    stop = entry - side * params.stop_atr * atr[i - 1]
                invalid = not (np.isfinite(entry) and entry > 0
                               and np.isfinite(stop) and np.isfinite(close[i]))
                stopped = side * (close[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid or stopped or expired:
                    side = 0
            elif candidate[i] and np.isfinite(close[i]) and close[i] > 0:
                side = int(candidate[i])
                entry_bar = i + 1
            signal[i] = side
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = SeasonalDebtParams()
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    for side in (1, -1):
        closes = [100, 108, 98, 96.5, 100, 100] if side == 1 else [100, 92, 102, 103.5, 100, 100]
        data = pd.DataFrame({"open": 100 + side, "close": closes}, index=index)
        derived = pd.DataFrame({
            "season_mean": side * 0.002, "volatility": 0.01,
            "debt": side * 0.01, "ma200": 100 - side * 20, "atr": 2.0
        }, index=index)
        result = strategy.generate_signals(data, derived, None, params).data
        assert result["signal"].iloc[:5].tolist() == [0, side, side, side, 0]
        assert result["size"].gt(0).all()
        timed = SeasonalDebtParams(hold_bars=2, stop_atr=100.0)
        result = strategy.generate_signals(data, derived, None, timed).data
        assert result["signal"].iloc[:4].tolist() == [0, side, side, 0]
    assert strategy.warmup_bars(params) == 505


if __name__ == "__main__":
    _self_check()
