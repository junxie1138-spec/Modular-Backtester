from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    entry_position: float = 0.15
    profit_target: float = 0.03


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = "gen_local_1791308125"

    @classmethod
    def params_type(cls) -> type[ElasticParams]:
        return ElasticParams

    def warmup_bars(self, params: ElasticParams) -> int:
        return 146

    def indicators(self, data: pd.DataFrame, params: ElasticParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["log_return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["displacement"] = out["log_return"].rolling(20, min_periods=20).sum()
        out["travel"] = out["log_return"].abs().rolling(20, min_periods=20).sum()
        out["efficiency"] = out["displacement"].abs().div(out["travel"].replace(0.0, np.nan))
        out["return_floor"] = out["displacement"].rolling(126, min_periods=126).min()
        out["return_ceiling"] = out["displacement"].rolling(126, min_periods=126).max()
        out["return_span"] = out["return_ceiling"] - out["return_floor"]
        out["relative_position"] = (out["displacement"] - out["return_floor"]).div(out["return_span"].replace(0.0, np.nan))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ElasticParams) -> SignalFrame:
        if not 0.0 < params.entry_position < 0.5:
            raise ValueError("entry_position must lie between zero and 0.5")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive")

        elastic = indicators["efficiency"].lt(0.35)
        buy = elastic & indicators["relative_position"].le(params.entry_position) & indicators["displacement"].lt(0.0) & indicators["log_return"].gt(0.0)
        sell = elastic & indicators["relative_position"].ge(1.0 - params.entry_position) & indicators["displacement"].gt(0.0) & indicators["log_return"].lt(0.0)
        candidates = np.select([buy.fillna(False), sell.fillna(False)], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        age = 0

        for i in range(len(data)):
            if position != 0:
                if age == 0:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        position = 0
                        continue
                age += 1
                gain = position * (closes[i] / entry_price - 1.0)
                if (np.isfinite(gain) and gain >= params.profit_target) or age >= 20:
                    position = 0
                    age = 0
                    entry_price = np.nan
                    continue
            elif candidates[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(candidates[i])
                age = 0
            decisions[i] = position

        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = ElasticParams()
    index = pd.date_range("2020-01-01", periods=25, freq="B")
    for side in (1, -1):
        data = pd.DataFrame({"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"efficiency": np.nan, "relative_position": np.nan, "displacement": np.nan, "log_return": np.nan}, index=index)
        ind.iloc[0] = [0.2, 0.0 if side == 1 else 1.0, -0.02 * side, 0.001 * side]
        signals = strategy.generate_signals(data, ind, None, params).data["signal"]
        assert signals.iloc[0] == 0
        assert (signals.iloc[1:21] == side).all()
        assert (signals.iloc[21:] == 0).all()
        data.loc[index[3], "close"] = 100.0 + 4.0 * side
        signals = strategy.generate_signals(data, ind, None, params).data["signal"]
        assert (signals.iloc[1:4] == side).all()
        assert (signals.iloc[4:] == 0).all()


if __name__ == "__main__":
    _self_check()
