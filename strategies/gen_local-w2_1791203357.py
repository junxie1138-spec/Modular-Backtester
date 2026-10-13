from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    correlation_window: int = 30
    acceleration_threshold: float = 0.15
    correlation_floor: float = 0.10
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791203357"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        if params.correlation_window < 3:
            raise ValueError("correlation_window must be at least 3")
        return int(params.correlation_window + 4)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        window = params.correlation_window
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["roc"] = close.pct_change(fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["intraday_return"] = close.div(opening).sub(1.0)
        for name in ("acceleration", "intraday_return"):
            series = out[name]
            history = series.rolling(window, min_periods=window)
            out[name + "_rho"] = history.corr(series.shift(1)).shift(1).clip(-1.0, 1.0)
            out[name + "_scale"] = history.std(ddof=0).shift(1).replace(0.0, np.nan)
            out[name + "_forecast"] = out[name + "_rho"] * series / out[name + "_scale"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if params.holding_bars not in (1, 2):
            raise ValueError("holding_bars must be 1 or 2")
        if not np.isfinite(params.acceleration_threshold) or params.acceleration_threshold <= 0:
            raise ValueError("acceleration_threshold must be finite and positive")
        if not np.isfinite(params.correlation_floor) or not 0 < params.correlation_floor <= 1:
            raise ValueError("correlation_floor must be in (0, 1]")
        acceleration = indicators["acceleration_forecast"]
        intraday = indicators["intraday_return_forecast"]
        valid = pd.Series(
            np.isfinite(indicators.to_numpy(dtype=float)).all(axis=1),
            index=data.index,
        )
        eligible = (
            valid
            & indicators["acceleration_rho"].abs().ge(params.correlation_floor)
            & indicators["intraday_return_rho"].abs().ge(params.correlation_floor)
        )
        long_entry = eligible & acceleration.ge(params.acceleration_threshold) & intraday.gt(0)
        short_entry = eligible & acceleration.le(-params.acceleration_threshold) & intraday.lt(0)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i, candidate in enumerate(candidates):
            if position != 0 and i >= exit_at:
                position = 0
            elif position == 0 and candidate != 0:
                position = int(candidate)
                exit_at = i + params.holding_bars
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = Params()
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame(index=index)
    forecasts = [1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    derived = pd.DataFrame({
        "acceleration_forecast": forecasts,
        "intraday_return_forecast": forecasts,
        "acceleration_rho": 0.5,
        "intraday_return_rho": 0.5,
    }, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result["size"].gt(0).all()
    derived["intraday_return_forecast"] = -derived["acceleration_forecast"]
    assert strategy.generate_signals(data, derived, None, params).data["signal"].eq(0).all()
    flat = pd.DataFrame({
        "open": 100.0, "high": 100.0, "low": 100.0,
        "close": 100.0, "volume": 1000.0,
    }, index=pd.date_range("2020-01-01", periods=100, freq="D"))
    flat_indicators = strategy.indicators(flat, params)
    assert flat_indicators.index.equals(flat.index)
    assert strategy.generate_signals(flat, flat_indicators, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
