from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticDiscountParams:
    ma_window: int = 40
    atr_window: int = 14
    vol_window: int = 20
    stability_window: int = 5
    stability_atr: float = 0.35
    discount_floor: float = -1.6
    discount_ceiling: float = -0.35
    target_vol: float = 0.12
    stop_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[ElasticDiscountParams]):
    strategy_id = "gen_local_1791252440"

    @classmethod
    def params_type(cls) -> type[ElasticDiscountParams]:
        return ElasticDiscountParams

    def warmup_bars(self, params: ElasticDiscountParams) -> int:
        return int(max(params.ma_window + params.stability_window,
                       params.atr_window + 1,
                       params.vol_window + 1) + 2)

    def indicators(self, data: pd.DataFrame, params: ElasticDiscountParams) -> pd.DataFrame:
        if min(params.ma_window, params.atr_window, params.vol_window,
               params.stability_window, params.holding_bars) < 1:
            raise ValueError("Lookbacks and holding_bars must be positive")
        if params.target_vol <= 0 or params.stop_atr <= 0 or params.stability_atr < 0:
            raise ValueError("Invalid volatility or stop parameters")
        if not params.discount_floor < params.discount_ceiling < 0:
            raise ValueError("Discount bounds must be ordered and negative")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ma"] = close.rolling(params.ma_window).mean()
        out["price_std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["price_std"].replace(0.0, np.nan)
        out["ma_displacement"] = out["ma"].diff(params.stability_window)
        out["equilibrium_drift"] = out["ma_displacement"].abs() / out["atr"].replace(0.0, np.nan)
        out["returns"] = close.pct_change(fill_method=None)
        out["annual_vol"] = out["returns"].rolling(params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out["size"] = (params.target_vol / out["annual_vol"].replace(0.0, np.nan)).clip(0.05, 0.95)
        out["relaxation"] = (
            out["z"].between(params.discount_floor, params.discount_ceiling)
            & out["z"].gt(out["z"].shift(1))
            & out["atr"].lt(out["atr"].shift(1))
            & out["equilibrium_drift"].le(params.stability_atr)
            & out["annual_vol"].gt(0.0)
            & np.isfinite(out["size"])
        ).fillna(False)
        out["confirmed"] = out["relaxation"] & out["relaxation"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ElasticDiscountParams) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_atr = np.nan
        fixed_stop = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_bar = i
                    fixed_stop = opens[i] - params.stop_atr * entry_atr
            if active:
                stop_fired = np.isfinite(closes[i]) and closes[i] < fixed_stop
                expired = i - entry_bar + 1 >= params.holding_bars
                if stop_fired or expired:
                    active = False
                else:
                    signals[i] = 1
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                entry_atr = atr[i]
                pending = True
                signals[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["size"].shift(1).replace([np.inf, -np.inf], np.nan).fillna(0.05).clip(0.05, 0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "close": [100.0, 100.0, 100.0, 94.0, 100.0, 100.0]}, index=index)
    indicators = pd.DataFrame({"atr": 2.0, "size": 0.5,
                               "confirmed": [False, True, False, False, False, False]}, index=index)
    params = ElasticDiscountParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    params.holding_bars = 2
    data["close"] = 100.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
