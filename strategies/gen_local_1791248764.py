from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    correlation_window: int = 30
    rank_window: int = 60
    atr_window: int = 14
    percentile: float = 0.85
    stop_atr: float = 1.25
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791248764"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if params.correlation_window < 3 or params.rank_window < 3 or params.atr_window < 1:
            raise ValueError("Invalid indicator window")
        if not 0.5 < params.percentile < 1.0:
            raise ValueError("percentile must be between 0.5 and 1")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        if params.hold_days not in (1, 2):
            raise ValueError("hold_days must be 1 or 2")
        return int(max(params.correlation_window + params.rank_window + 3, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        returns = out["return"]
        # Fit through the previous close; apply those coefficients to today's returns.
        out["rho_one"] = returns.rolling(params.correlation_window).corr(returns.shift(1)).shift(1)
        out["rho_two"] = returns.rolling(params.correlation_window).corr(returns.shift(2)).shift(1)
        out["forecast"] = out["rho_one"] * returns + out["rho_two"] * returns.shift(1)
        out["forecast_rank"] = out["forecast"].rolling(params.rank_window).rank(pct=True)
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["pressure"] = (data["close"] - data["open"]) / out["range"]
        out["pressure_rank"] = out["pressure"].rolling(params.rank_window).rank(pct=True)
        out["previous_close"] = data["close"].shift(1)
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = pd.concat([out["range"], out["high_gap"], out["low_gap"]], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        self.warmup_bars(params)
        upper = params.percentile
        lower = 1.0 - upper
        forecast_long = (indicators["forecast_rank"] >= upper) & (indicators["forecast"] > 0)
        forecast_short = (indicators["forecast_rank"] <= lower) & (indicators["forecast"] < 0)
        pressure_long = (indicators["pressure_rank"] >= upper) & (indicators["pressure"] > 0)
        pressure_short = (indicators["pressure_rank"] <= lower) & (indicators["pressure"] < 0)
        candidate = np.select([forecast_long & pressure_long, forecast_short & pressure_short], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        stop = np.nan
        frozen_atr = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    # Entry fills at this open; ATR was frozen at the decision close.
                    stop = opens[i] - position * params.stop_atr * frozen_atr
                stopped = not np.isfinite(closes[i]) or position * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.hold_days
                if stopped or expired:
                    position = 0
            elif candidate[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidate[i])
                entry_bar = i + 1
                frozen_atr = atr[i]
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = Params()
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"forecast": [0.01, 0.0, 0.0, 0.0], "forecast_rank": [0.95, 0.5, 0.5, 0.5], "pressure": [0.8, 0.0, 0.0, 0.0], "pressure_rank": [0.95, 0.5, 0.5, 0.5], "atr": 1.0}, index=index)
    assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [0, 1, 1, 0]
    data.loc[index[1], "close"] = 98.0
    assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [0, 1, 0, 0]
    ind.loc[index[0], ["forecast", "forecast_rank", "pressure", "pressure_rank"]] = [-0.01, 0.05, -0.8, 0.05]
    data.loc[index[1], "close"] = 102.0
    assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [0, -1, 0, 0]
    ind.loc[index[0], "pressure_rank"] = 0.95
    assert strategy.generate_signals(data, ind, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
