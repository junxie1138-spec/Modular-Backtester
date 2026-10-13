from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    volume_lookback: int = 20
    volume_multiple: float = 1.25
    min_intraday_return: float = 0.002
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791196449"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if params.volume_lookback < 1 or params.hold_bars not in (1, 2):
            raise ValueError("Positive volume lookback and one- or two-bar holding period required.")
        if not np.isfinite(params.volume_multiple) or params.volume_multiple <= 0:
            raise ValueError("volume_multiple must be finite and positive.")
        if not np.isfinite(params.min_intraday_return) or params.min_intraday_return < 0:
            raise ValueError("min_intraday_return must be finite and nonnegative.")
        return int(params.volume_lookback + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["day_of_month"] = data.index.day
        out["funding_window"] = (
            out["day_of_month"].between(1, 3)
            | out["day_of_month"].between(15, 17)
        )
        out["intraday_return"] = (
            data["close"].where(data["close"] > 0)
            / data["open"].where(data["open"] > 0) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        out["prior_volume_mean"] = (
            data["volume"].where(data["volume"] > 0)
            .rolling(params.volume_lookback, min_periods=params.volume_lookback)
            .mean().shift(1)
        )
        out["relative_volume"] = (
            data["volume"].where(data["volume"] > 0)
            / out["prior_volume_mean"].where(out["prior_volume_mean"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["volume_confirmed_advance"] = (
            (out["intraday_return"] > params.min_intraday_return)
            & (out["relative_volume"] >= params.volume_multiple)
        ).fillna(False)
        out["entry"] = (
            out["funding_window"] & out["volume_confirmed_advance"]
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2.")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, entry in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Exit takes precedence; entries cannot extend the holding clock.
                    exit_bar = -1
            elif entry:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2024-01-01", periods=7, freq="B")
    data = pd.DataFrame(index=index)
    for hold, expected in (
        (1, [0, 1, 0, 1, 0, 1, 0]),
        (2, [0, 1, 1, 0, 1, 1, 0]),
    ):
        indicators = pd.DataFrame({"entry": True}, index=index)
        result = GeneratedStrategy.generate_signals(
            None, data, indicators, None, Params(hold_bars=hold)
        ).data
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()
    indicators = pd.DataFrame(
        {"entry": [True, False, False, False, False, False, False]}, index=index
    )
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, Params()
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
