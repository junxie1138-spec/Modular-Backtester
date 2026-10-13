from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 6
    correlation_window: int = 20
    area_threshold: float = 0.04
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791238040"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(200, params.signature_window + 2, params.correlation_window + 2))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.signature_window < 3 or params.correlation_window < 3:
            raise ValueError("Windows must be at least three bars")
        if params.area_threshold <= 0 or params.profit_target <= 0:
            raise ValueError("Area threshold and profit target must be positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two bars")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lag_return"] = out["return"].shift(1)
        x = out["lag_return"]
        y = out["return"]
        window = params.signature_window
        # Second-order antisymmetric signature of cumulative (lagged, current) returns.
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, window):
            cross = x.shift(lag) * y - y.shift(lag) * x
            area = area + 0.5 * cross.rolling(window - lag, min_periods=window - lag).sum()
        out["signature_area"] = area
        out["path_length"] = (x.abs() + y.abs()).rolling(window, min_periods=window).sum()
        out["normalized_area"] = area.div(out["path_length"].pow(2).replace(0.0, np.nan))
        out["autocorrelation"] = y.rolling(params.correlation_window, min_periods=params.correlation_window).corr(x)
        out["ma_200"] = close.rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        if params.profit_target <= 0 or params.max_hold_bars not in (1, 2):
            raise ValueError("Require a positive target and a one- or two-bar holding horizon")
        setup = (
            (data["close"] > indicators["ma_200"])
            & (indicators["normalized_area"] < -params.area_threshold)
            & (indicators["autocorrelation"] < 0)
            & (indicators["lag_return"] < 0)
            & (indicators["return"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        pending = False
        entry_price = np.nan
        held_bars = 0
        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i]
                    held_bars = 0
                else:
                    continue
            if np.isfinite(entry_price):
                held_bars += 1
                reached_target = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if reached_target or held_bars >= params.max_hold_bars:
                    entry_price = np.nan
                    held_bars = 0
                    # A zero decision exits at the following open.
                    continue
                decisions[i] = 1
            elif setup[i]:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"ma_200": 99.0, "normalized_area": [-0.1, 0.0, 0.0, 0.0, 0.0], "autocorrelation": -0.2, "lag_return": -0.01, "return": 0.01}, index=index)
    strategy = GeneratedStrategy()
    params = SignatureParams()
    timed = strategy.generate_signals(data, derived, None, params).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 101.0
    targeted = strategy.generate_signals(data, derived, None, params).data
    assert targeted["signal"].tolist() == [0, 1, 0, 0, 0]
    assert targeted["size"].gt(0).all()
    assert strategy.indicators(data, params).index.equals(data.index)
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
