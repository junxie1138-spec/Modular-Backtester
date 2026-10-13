from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpirationPressureParams:
    rank_window: int = 8
    pressure_percentile: float = 0.875
    holding_bars: int = 18

    def __post_init__(self):
        if not 2 <= self.rank_window <= 10:
            raise ValueError("rank_window must be between 2 and 10")
        if not 0.0 < self.pressure_percentile <= 1.0:
            raise ValueError("pressure_percentile must be in (0, 1]")
        if not 15 <= self.holding_bars <= 20:
            raise ValueError("holding_bars must be between 15 and 20")


class GeneratedStrategy(BaseStrategy[ExpirationPressureParams]):
    strategy_id = "gen_local-w2_1791202721"

    @classmethod
    def params_type(cls):
        return ExpirationPressureParams

    def warmup_bars(self, params: ExpirationPressureParams) -> int:
        return int(params.rank_window)

    def indicators(self, data: pd.DataFrame, params: ExpirationPressureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["session_range"] = (data["high"] - data["low"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        valid = (
            np.isfinite(data[["open", "high", "low", "close", "volume"]]).all(axis=1)
            & out["session_range"].gt(0.0)
            & data["volume"].gt(0.0)
            & data["open"].between(data["low"], data["high"])
            & data["close"].between(data["low"], data["high"])
        )
        out["valid_bar"] = valid
        pressure = (
            data["volume"] * (data["open"] - data["close"])
            / out["session_range"].where(valid)
        ).replace([np.inf, -np.inf], np.nan)
        out["selling_pressure"] = pressure.fillna(0.0)
        # Rolling percentile API: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.typing.Rolling.rank.html
        out["pressure_rank"] = pressure.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method="average", pct=True).fillna(0.0)
        out["expiration_friday"] = (
            (data.index.dayofweek == 4)
            & (data.index.day >= 15)
            & (data.index.day <= 21)
        )
        out["entry"] = (
            out["valid_bar"]
            & out["expiration_friday"]
            & out["selling_pressure"].gt(0.0)
            & out["pressure_rank"].ge(params.pressure_percentile)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ExpirationPressureParams,
    ) -> SignalFrame:
        positions = np.zeros(len(data), dtype=int)
        exit_at = -1
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i, entry in enumerate(entries):
            if exit_at >= 0:
                if i < exit_at:
                    positions[i] = 1
                else:
                    # Force the exit even if a fresh entry condition is true.
                    exit_at = -1
            elif entry:
                positions[i] = 1
                exit_at = i + params.holding_bars
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    for holding_bars in (15, 18, 20):
        params = ExpirationPressureParams(holding_bars=holding_bars)
        index = pd.bdate_range("2020-01-01", periods=65)
        data = pd.DataFrame(index=index)
        indicators = pd.DataFrame({"entry": True}, index=index)
        signal = strategy.generate_signals(data, indicators, None, params).data["signal"]
        assert signal.iloc[0] == 0
        assert signal.iloc[1:holding_bars + 1].eq(1).all()
        assert signal.iloc[holding_bars + 1] == 0
        assert signal.iloc[holding_bars + 2:2 * holding_bars + 2].eq(1).all()
        assert signal.iloc[2 * holding_bars + 2] == 0
        assert strategy.warmup_bars(params) <= 10


if __name__ == "__main__":
    _self_check()
