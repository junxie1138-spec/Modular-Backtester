from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReservoirParams:
    roc_days: int = 3
    reservoir_days: int = 5
    vol_days: int = 20
    atr_days: int = 14
    entry_pressure: float = 2.0
    target_vol: float = 0.14
    stop_atr: float = 2.0
    hold_days: int = 8


class GeneratedStrategy(BaseStrategy[ReservoirParams]):
    strategy_id = "gen_local-w2_1791322935"

    @classmethod
    def params_type(cls):
        return ReservoirParams

    def warmup_bars(self, params: ReservoirParams) -> int:
        return max(200, params.atr_days + 1,
                   max(params.vol_days + 1, params.roc_days + 2)
                   + params.reservoir_days)

    def indicators(self, data: pd.DataFrame, params: ReservoirParams) -> pd.DataFrame:
        if min(params.roc_days, params.reservoir_days, params.vol_days,
               params.atr_days, params.hold_days) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if params.vol_days < 2 or not all(
            np.isfinite(x) and x > 0
            for x in (params.entry_pressure, params.target_vol, params.stop_atr)
        ):
            raise ValueError("Require vol_days >= 2 and finite positive risk parameters.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["daily_vol"] = out["return"].rolling(params.vol_days).std(ddof=0)
        out["annual_vol"] = out["daily_vol"] * np.sqrt(252.0)
        out["roc"] = close.pct_change(params.roc_days, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["impulse"] = (
            out["acceleration"] / (np.sqrt(2.0) * out["daily_vol"].replace(0, np.nan))
        ).clip(-1.0, 1.0)
        out["pressure"] = out["impulse"].rolling(params.reservoir_days).sum()
        out["previous_pressure"] = out["pressure"].shift(1)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_days).mean()
        out["ma200"] = close.rolling(200).mean()
        out["target_size"] = (
            params.target_vol / out["annual_vol"].replace(0, np.nan)
        ).clip(0.01, 1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ReservoirParams,
    ) -> SignalFrame:
        ready = np.isfinite(indicators[
            ["roc", "pressure", "previous_pressure", "atr", "ma200", "annual_vol"]
        ]).all(axis=1)
        ready &= (indicators["atr"] > 0) & (indicators["annual_vol"] > 0)
        ready &= np.isfinite(data["close"]) & (data["close"] > 0)
        regime = data["close"] > indicators["ma200"]
        enter = (
            ready & regime & (indicators["roc"] > 0)
            & (indicators["pressure"] >= params.entry_pressure)
            & (indicators["previous_pressure"] < params.entry_pressure)
        ).fillna(False).to_numpy()
        eligible = (ready & regime).fillna(False).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_bar = -1
        entry_atr = 0.0
        stop = 0.0
        for i in range(len(data)):
            if pending:
                pending = False
                active = np.isfinite(opens[i]) and opens[i] > 0
                entry_bar = i
                # Anchor once to the fill-bar open and decision-bar ATR.
                stop = opens[i] - params.stop_atr * entry_atr
            if active:
                if (not eligible[i] or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_days):
                    active = False
                else:
                    raw[i] = 1
            elif enter[i]:
                pending = True
                entry_atr = atr[i]
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = indicators["target_size"].shift(1).fillna(0.01).clip(0.01, 1.0).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    idx = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [110., 120., 119., 125., 119., 117.],
        "high": [111., 121., 126., 126., 120., 118.],
        "low": [109., 118., 118., 118., 116., 109.],
        "close": [110., 119., 125., 119., 117., 110.],
        "volume": [1000.] * 6,
    }, index=idx)
    ind = pd.DataFrame({
        "roc": [0.01] * 6, "pressure": [2.1] * 6,
        "previous_pressure": [1.9] + [2.1] * 5,
        "atr": [1.] * 6, "ma200": [100.] * 6,
        "annual_vol": [0.2] * 6, "target_size": [0.7] * 6,
    }, index=idx)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, ind, ctx, ReservoirParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    assert np.allclose(result["size"].iloc[1:], 0.7)
    short_hold = strategy.generate_signals(
        data, ind, ctx, ReservoirParams(hold_days=2)
    ).data
    assert short_hold["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    ind["ma200"] = 200.
    assert not strategy.generate_signals(data, ind, ctx, ReservoirParams()).data["signal"].any()


if __name__ == "__main__":
    _self_check()
