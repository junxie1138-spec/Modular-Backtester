from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ImpactParams:
    atr_window: int = 14
    regime_window: int = 20
    history_window: int = 126
    event_percentile: float = 0.80
    regime_percentile: float = 0.65
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[ImpactParams]):
    strategy_id = "gen_local-w2_1791195985"

    @classmethod
    def params_type(cls) -> type[ImpactParams]:
        return ImpactParams

    @staticmethod
    def warmup_bars(params: ImpactParams) -> int:
        if min(params.atr_window, params.regime_window, params.history_window) < 2:
            raise ValueError("Lookback windows must be at least two.")
        if not (0.5 < params.event_percentile < 1.0 and 0.5 < params.regime_percentile < 1.0):
            raise ValueError("Percentiles must lie strictly between 0.5 and 1.")
        if not (params.breakeven_pct > 0 and params.trail_atr > 0 and params.max_hold_bars >= 1):
            raise ValueError("Exit parameters must be positive.")
        return int(params.atr_window + params.regime_window + params.history_window + 2)

    def indicators(self, data: pd.DataFrame, params: ImpactParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window).mean()
        out["move"] = data["close"].diff().div(out["atr"].shift(1).replace(0.0, np.nan))
        out["impact"] = out["move"].abs()
        out["log_volume"] = np.log1p(data["volume"].clip(lower=0))
        out["coupling"] = out["impact"].rolling(params.regime_window).corr(out["log_volume"])
        history = params.history_window
        out["volume_cutoff"] = data["volume"].rolling(history).quantile(params.event_percentile).shift(1)
        out["impact_cutoff"] = out["impact"].rolling(history).quantile(params.event_percentile).shift(1)
        out["follow_cutoff"] = out["coupling"].rolling(history).quantile(params.regime_percentile).shift(1)
        out["fade_cutoff"] = out["coupling"].rolling(history).quantile(1.0 - params.regime_percentile).shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out["regime"] = np.select([
            ready & (out["coupling"] > out["follow_cutoff"]),
            ready & (out["coupling"] < out["fade_cutoff"]),
        ], [1, -1], default=0).astype(int)
        confirmed = ready & (data["volume"] > out["volume_cutoff"]) & (out["impact"] > out["impact_cutoff"])
        out["trigger"] = np.where(confirmed, np.sign(out["move"].fillna(0)) * out["regime"], 0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ImpactParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        triggers = indicators["trigger"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry, stop = np.nan, -np.inf
        armed = False
        for i, (opening, high, low, closing) in enumerate(prices):
            if not (np.isfinite(prices[i]).all() and min(opening, high, low, closing) > 0
                    and np.isfinite(atr[i]) and atr[i] > 0):
                side, entry, armed = 0, np.nan, False
                continue
            if side:
                if np.isnan(entry):
                    # The previous close's command fills at this bar's open.
                    entry, age = opening, 0
                    stop = side * entry - params.trail_atr * atr[i - 1]
                age += 1
                worst = low if side > 0 else -high
                # Detect the previously established stop; exit executes next open.
                if worst <= stop or age >= params.max_hold_bars:
                    side, entry, armed = 0, np.nan, False
                    continue
                favorable = side * closing
                armed = armed or favorable - side * entry >= entry * params.breakeven_pct
                if armed:
                    # Signed-price stops only increase, including for shorts.
                    stop = max(stop, side * entry, favorable - params.trail_atr * atr[i])
            else:
                side = int(triggers[i])
                entry, armed = np.nan, False
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    data = pd.DataFrame(
        [
            [100, 101, 99, 100], [101, 104, 101, 103], [103, 105, 102, 104],
            [103, 104, 101, 103], [100, 101, 99, 100], [100, 100, 97, 98],
            [98, 99, 96, 97], [98, 100, 97, 99], [99, 100, 98, 99],
        ],
        columns=["open", "high", "low", "close"],
        index=pd.date_range("2020-01-01", periods=9),
    )
    data["volume"] = 1000.0
    supplied = pd.DataFrame({"atr": 1.0, "trigger": [1, 0, 0, 0, -1, 0, 0, 0, 0]}, index=data.index)
    strategy, params = GeneratedStrategy(), ImpactParams()
    result = strategy.generate_signals(data, supplied, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert (result["size"] > 0).all()
    for end in range(1, len(data)):
        partial = strategy.generate_signals(data.iloc[:end], supplied.iloc[:end], None, params).data
        assert partial.equals(result.iloc[:end])


if __name__ == "__main__":
    _self_check()
