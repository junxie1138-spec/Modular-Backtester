from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    window: int = 6
    breadth_gain: float = 1.0
    volume_threshold: float = 0.15
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791287937"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        if params.window < 2 or params.hold_bars not in (1, 2):
            raise ValueError("window must be >= 2 and hold_bars must be 1 or 2")
        if not np.isfinite(params.breadth_gain) or params.breadth_gain <= 0:
            raise ValueError("breadth_gain must be positive and finite")
        if not np.isfinite(params.volume_threshold) or not 0 <= params.volume_threshold < 1:
            raise ValueError("volume_threshold must be in [0, 1)")
        return 2 * params.window + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        w = params.window
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        upside = returns.clip(lower=0)
        upside_sum = upside.rolling(w, min_periods=w).sum()
        upside_energy = upside.pow(2).rolling(w, min_periods=w).sum()
        breadth = upside_sum.pow(2).div(upside_energy.where(upside_energy > 0))
        volume = data["volume"].where(np.isfinite(data["volume"]) & (data["volume"] >= 0))
        signed_volume = volume * np.sign(returns)
        volume_total = volume.rolling(w, min_periods=w).sum()
        balance = signed_volume.rolling(w, min_periods=w).sum().div(volume_total.where(volume_total > 0))
        return pd.DataFrame({
            "return": returns,
            "net_return": returns.rolling(w, min_periods=w).sum(),
            "upside_breadth": breadth,
            "previous_breadth": breadth.shift(w),
            "volume_balance": balance,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        warmup = GeneratedStrategy.warmup_bars(params)
        ready = np.isfinite(indicators).all(axis=1)
        momentum = (
            (indicators["return"] > 0)
            & (indicators["net_return"] > 0)
            & (indicators["upside_breadth"] >= indicators["previous_breadth"] + params.breadth_gain)
        )
        participation = indicators["volume_balance"] >= params.volume_threshold
        eligible = (ready & momentum & participation).to_numpy(dtype=bool)
        eligible[:warmup - 1] = False
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
            elif eligible[i]:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    forced = pd.DataFrame({"return": 0.01, "net_return": 0.02, "upside_breadth": 3.0, "previous_breadth": 1.0, "volume_balance": 0.5}, index=index)
    for hold, expected in ((1, [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]), (2, [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1])):
        params = Params(window=2, hold_bars=hold)
        result = GeneratedStrategy.generate_signals(data, forced, None, params).data
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()
    actual = GeneratedStrategy.indicators(data, Params(window=2))
    assert GeneratedStrategy.generate_signals(data, actual, None, Params(window=2)).data["signal"].eq(0).all()
    assert GeneratedStrategy.generate_signals(data.iloc[:0], actual.iloc[:0], None, Params(window=2)).data.empty


if __name__ == "__main__":
    _self_check()
