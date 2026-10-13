from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityPolarityParams:
    downside_share: float = 0.30
    stop_atr: float = 1.50


class GeneratedStrategy(BaseStrategy[VolatilityPolarityParams]):
    strategy_id = "gen_local-w2_1791272933"

    @classmethod
    def params_type(cls):
        return VolatilityPolarityParams

    def warmup_bars(self, params: VolatilityPolarityParams) -> int:
        return 15

    def indicators(self, data: pd.DataFrame, params: VolatilityPolarityParams) -> pd.DataFrame:
        if not 0.0 < params.downside_share < 0.5:
            raise ValueError("downside_share must be between zero and 0.5")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["change"] = data["close"].diff()
        out["up_squared"] = out["change"].clip(lower=0.0).pow(2)
        out["down_squared"] = out["change"].clip(upper=0.0).pow(2)
        out["up_vol"] = np.sqrt(out["up_squared"].rolling(12, min_periods=12).mean())
        out["down_vol"] = np.sqrt(out["down_squared"].rolling(12, min_periods=12).mean())
        out["total_vol"] = out["up_vol"] + out["down_vol"]
        out["downside_share"] = out["down_vol"] / out["total_vol"].replace(0.0, np.nan)
        out["displacement"] = out["change"].rolling(12, min_periods=12).sum()
        out["noise"] = np.sqrt((out["up_squared"] + out["down_squared"]).rolling(12, min_periods=12).sum())
        out["strength"] = out["displacement"] / out["noise"].replace(0.0, np.nan)
        out["range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["range", "high_gap", "low_gap"]].max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["previous_share"] = out["downside_share"].shift(1)
        out["entry"] = (
            (out["downside_share"] < params.downside_share)
            & (out["previous_share"] >= params.downside_share)
            & (out["strength"] > 1.5)
            & (out["atr"] > 0.0)
            & np.isfinite(out[["downside_share", "previous_share", "strength", "atr"]]).all(axis=1)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolatilityPolarityParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        stop = np.nan
        frozen_atr = np.nan
        age = 0
        for i in range(len(data)):
            if active:
                if pending:
                    # Freeze the stop at the actual entry bar's open, using decision-bar ATR.
                    stop = opens[i] - params.stop_atr * frozen_atr
                    pending = False
                age += 1
                if (
                    not np.isfinite(stop)
                    or not np.isfinite(closes[i])
                    or closes[i] < stop
                    or age >= 4
                ):
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                active = True
                pending = True
                frozen_atr = atr[i]
                age = 0
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"atr": 1.0, "entry": [True] + [False] * 7}, index=index)
    strategy = GeneratedStrategy()
    params = VolatilityPolarityParams(stop_atr=2.0)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=15)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[1], ["open", "close"]] = [110.0, 109.0]
    data.loc[index[2], "close"] = 107.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    computed = strategy.indicators(data, params)
    assert not computed["entry"].any()
    assert strategy.warmup_bars(params) >= 15
    assert result["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
