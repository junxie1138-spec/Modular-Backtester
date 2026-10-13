from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    z_threshold: float = 1.25
    impact_ratio: float = 0.5
    target_vol: float = 0.10
    stop_atr: float = 1.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791274903"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if params.lookback < 2 or params.hold_bars not in (1, 2):
            raise ValueError("lookback >= 2 and hold_bars in {1, 2} required")
        values = (params.z_threshold, params.impact_ratio,
                  params.target_vol, params.stop_atr)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("thresholds, target_vol and stop_atr must be positive")
        if params.impact_ratio >= 1:
            raise ValueError("impact_ratio must be below 1")
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        n = params.lookback
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(n, min_periods=n).mean()
        out["price_std"] = close.rolling(n, min_periods=n).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["price_std"].replace(0, np.nan)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(n, min_periods=n).mean()
        out["return"] = close.pct_change(fill_method=None)
        out["realized_vol"] = (
            out["return"].rolling(n, min_periods=n).std(ddof=0) * np.sqrt(252.0)
        )
        out["mean_volume"] = data["volume"].rolling(n, min_periods=n).mean()
        out["relative_volume"] = (
            data["volume"] / out["mean_volume"].replace(0, np.nan)
        )
        out["sell_impact"] = (
            (data["open"] - close).clip(lower=0)
            / (out["atr"] * out["relative_volume"]).replace(0, np.nan)
        )
        out["previous_impact"] = out["sell_impact"].shift(1)
        out["vol_size"] = (
            params.target_vol / out["realized_vol"].replace(0, np.nan)
        ).clip(lower=0.01, upper=0.95)
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = (
            out[["z", "atr", "realized_vol", "sell_impact",
                 "previous_impact", "vol_size"]].notna().all(axis=1)
            & (out["atr"] > 0) & (out["realized_vol"] > 0)
            & (data["volume"] > 0) & (close > 0)
        )
        out["setup"] = (
            valid & (out["z"] <= -params.z_threshold)
            & (out["sell_impact"] > 0)
            & (out["previous_impact"] > 0)
            & (out["sell_impact"] <= params.impact_ratio * out["previous_impact"])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        self.warmup_bars(params)
        df = pd.DataFrame({"signal": 0, "size": 0.01}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.01, dtype=float)
        active = False
        decision_bar = -1
        frozen_atr = 0.0
        fixed_stop = np.nan
        position_size = 0.01
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            if active:
                if i == decision_bar + 1:
                    entry_open = float(data["open"].iloc[i])
                    # The next open is observed only on the fill bar.
                    fixed_stop = entry_open - params.stop_atr * frozen_atr
                    if not np.isfinite(entry_open) or entry_open <= 0:
                        active = False
                age = i - decision_bar
                if (not active or not np.isfinite(close) or close <= 0
                        or close < fixed_stop or age >= params.hold_bars):
                    active = False
                    continue
                signals[i] = 1
                sizes[i] = position_size
            elif bool(indicators["setup"].iloc[i]):
                active = True
                decision_bar = i
                frozen_atr = float(indicators["atr"].iloc[i])
                position_size = float(indicators["vol_size"].iloc[i])
                signals[i] = 1
                sizes[i] = position_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame(
        {"open": [100.0] * 4, "high": [106.0] * 4,
         "low": [97.0] * 4, "close": [100.0, 99.0, 100.0, 100.0],
         "volume": [1000.0] * 4}, index=index,
    )
    ind = pd.DataFrame(
        {"setup": [True, False, False, False],
         "atr": [1.0, 100.0, 100.0, 100.0], "vol_size": [0.4] * 4},
        index=index,
    )
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    assert result["size"].iloc[1:3].tolist() == [0.4, 0.4]
    data.loc[index[1], "close"] = 98.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0]
    ind["setup"] = False
    assert not strategy.generate_signals(data, ind, None, params).data["signal"].any()
    constant = data.copy()
    constant[["open", "high", "low", "close"]] = 100.0
    derived = strategy.indicators(constant, params)
    assert not derived["setup"].any()
    assert strategy.warmup_bars(params) == 21


if __name__ == "__main__":
    _self_check()
