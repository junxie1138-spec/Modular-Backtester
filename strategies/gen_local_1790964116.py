from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnBudgetParams:
    return_window: int = 6
    shock_multiple: float = 2.0
    balance_limit: float = 0.25
    profit_target: float = 0.009
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[ReturnBudgetParams]):
    strategy_id = "gen_local_1790964116"

    @classmethod
    def params_type(cls) -> type[ReturnBudgetParams]:
        return ReturnBudgetParams

    @staticmethod
    def warmup_bars(params: ReturnBudgetParams) -> int:
        return max(3, min(8, int(params.return_window))) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: ReturnBudgetParams) -> pd.DataFrame:
        window = max(3, min(8, int(params.return_window)))
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        prior = returns.shift(1)
        gross = prior.abs().rolling(window, min_periods=window).sum()
        net = prior.rolling(window, min_periods=window).sum()
        scale = gross / window
        balance = net.abs() / gross.replace(0.0, np.nan)
        ready = returns.notna() & scale.notna() & balance.notna() & scale.gt(0.0)
        isolated = ready & balance.le(max(0.0, min(1.0, params.balance_limit)))
        isolated &= returns.abs().ge(max(1.0, params.shock_multiple) * scale)
        entry = pd.Series(0, index=data.index, dtype=int)
        entry.loc[isolated & returns.lt(0.0)] = 1
        entry.loc[isolated & returns.gt(0.0)] = -1
        return pd.DataFrame({
            "return": returns.fillna(0.0),
            "prior_gross": gross.fillna(0.0),
            "prior_net": net.fillna(0.0),
            "prior_scale": scale.fillna(0.0),
            "prior_balance": balance.fillna(0.0),
            "ready": ready,
            "entry": entry,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ReturnBudgetParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = 0.0
        entry_bar = -1
        target = max(0.0001, float(params.profit_target))
        hold = max(3, min(5, int(params.max_hold)))

        for i in range(len(data)):
            # A preceding close decision fills at this session's open.
            if position != 0 and entry_bar == -1:
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    entry_price = opens[i]
                    entry_bar = i
                else:
                    position = 0

            if position != 0:
                gain = position * (closes[i] / entry_price - 1.0)
                profit_exit = np.isfinite(gain) and gain >= target
                time_exit = i - entry_bar + 1 >= hold
                if profit_exit or time_exit:
                    position = 0
                    entry_bar = -1
            elif entries[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(entries[i])
                entry_bar = -1

            decisions[i] = position

        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100.0] * 7,
        "high": [103.0] * 7,
        "low": [97.0] * 7,
        "close": [100.0, 100.0, 101.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 7,
    }, index=index)
    indicators = pd.DataFrame({"entry": [1, 0, 0, -1, 0, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, ReturnBudgetParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, -1]
    flat = data.copy()
    flat["close"] = 100.0
    result = GeneratedStrategy.generate_signals(flat, indicators, None, ReturnBudgetParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    derived = GeneratedStrategy.indicators(flat, ReturnBudgetParams())
    assert not derived.isna().any().any()
    assert derived["entry"].eq(0).all()
    assert GeneratedStrategy.warmup_bars(ReturnBudgetParams(return_window=8)) == 10


if __name__ == "__main__":
    _self_check()
