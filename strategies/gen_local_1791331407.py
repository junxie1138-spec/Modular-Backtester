from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    window: int = 8
    profit_target: float = 0.008


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791331407"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return max(2, int(params.window)) + 1

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.window < 2 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("window must be >= 2 and profit_target must be positive")
        n = int(params.window)
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).where(lambda s: s > 0)
        opening = data["open"].where(lambda s: s > 0)
        out["gap"] = (opening / previous - 1).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = (data["close"].where(lambda s: s > 0) / opening - 1).replace([np.inf, -np.inf], np.nan)
        x, y = out["gap"], out["intraday"]
        area = pd.Series(0.0, index=data.index)
        # Second-order antisymmetric signature of the cumulative gap/intraday path.
        for lag in range(1, n):
            cross = x.shift(lag) * y - y.shift(lag) * x
            area = area + cross.rolling(n - lag, min_periods=n - lag).sum()
        out["variation"] = (x.abs() + y.abs()).rolling(n, min_periods=n).sum()
        out["displacement"] = (x + y).rolling(n, min_periods=n).sum()
        scale = x.abs().rolling(n, min_periods=n).sum() * y.abs().rolling(n, min_periods=n).sum()
        out["signature_area"] = (0.5 * area / scale.where(scale > 0)).fillna(0.0)
        out["efficiency"] = (out["displacement"].abs() / out["variation"].where(out["variation"] > 0)).fillna(0.0)
        out["ready"] = out["displacement"].notna() & scale.gt(0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive")
        direction = np.sign(indicators["displacement"].fillna(0.0))
        persistent = indicators["efficiency"].ge(0.35) & indicators["signature_area"].gt(0.12)
        recycling = indicators["efficiency"].lt(0.35) & indicators["signature_area"].lt(-0.12)
        candidate = pd.Series(np.where(persistent, direction, np.where(recycling, -direction, 0)), index=data.index)
        candidate = candidate.where(indicators["ready"].fillna(False), 0).astype(int).to_numpy()
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if side:
                # The previous close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opening[i]
                if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(closing[i]) or closing[i] <= 0:
                    side = 0
                else:
                    gain = side * (closing[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= 2:
                        side = 0
                # An exit occupies this decision bar; no immediate re-entry.
            elif candidate[i]:
                side = int(candidate[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"displacement": 1.0, "efficiency": 0.5, "signature_area": 0.2, "ready": True}, index=index)
    strategy = GeneratedStrategy()
    def signals(prices, derived, target):
        return strategy.generate_signals(prices, derived, None, SignatureParams(profit_target=target)).data["signal"].tolist()
    assert signals(data, ind, 0.5) == [0, 1, 1, 0, 1, 1, 0]
    data["close"] = 101.0
    assert signals(data, ind, 0.005) == [0, 1, 0, 1, 0, 1, 0]
    ind["displacement"] = -1.0
    data["close"] = 99.0
    assert signals(data, ind, 0.005) == [0, -1, 0, -1, 0, -1, 0]
    ind["efficiency"], ind["signature_area"] = 0.2, -0.2
    data["close"] = 100.0
    assert signals(data, ind, 0.5) == [0, 1, 1, 0, 1, 1, 0]
    params = SignatureParams()
    derived = strategy.indicators(data, params)
    assert not derived["ready"].any()
    assert signals(data, derived, 0.5) == [0] * len(data)
    assert strategy.warmup_bars(params) >= params.window + 1


if __name__ == "__main__":
    _self_check()
