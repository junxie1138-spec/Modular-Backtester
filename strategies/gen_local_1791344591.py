from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fast_window: int = 12
    slow_window: int = 36
    vol_window: int = 20
    min_contraction: float = 0.035
    min_up_bias: float = 0.08
    annual_vol_target: float = 0.12
    profit_target: float = 0.008
    max_hold_bars: int = 2
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791344591"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.fast_window + 3, params.slow_window + 3,
                       params.vol_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (2 <= params.fast_window < params.slow_window
                and params.vol_window >= 2
                and 0 < params.min_contraction < 1
                and 0 <= params.min_up_bias < 1
                and params.annual_vol_target > 0
                and params.profit_target > 0
                and params.max_hold_bars in (1, 2)
                and 0 < params.max_size <= 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        a, b, c = out["return"].shift(2), out["return"].shift(1), out["return"]
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        # Pairwise order encodes six permutations; ties favor earlier observations.
        code = (a > b).astype(int) + 2 * (a > c).astype(int) + 4 * (b > c).astype(int)
        out["ordinal_code"] = code.where(valid)
        for label, window in (("fast", params.fast_window), ("slow", params.slow_window)):
            entropy = pd.Series(0.0, index=data.index)
            for pattern in (0, 1, 3, 4, 6, 7):
                probability = out["ordinal_code"].eq(pattern).astype(float).where(valid)
                probability = probability.rolling(window, min_periods=window).mean()
                out[f"p_{label}_{pattern}"] = probability
                entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
            out[f"entropy_{label}"] = entropy / np.log(6.0)
        out["contraction"] = out["entropy_slow"] - out["entropy_fast"]
        out["up_bias"] = out["p_fast_0"] - out["p_fast_7"]
        out["strength"] = (
            (out["contraction"] / 0.20).clip(0, 1)
            * (out["up_bias"] / 0.25).clip(0, 1)
        ).fillna(0.0)
        out["annual_vol"] = out["return"].rolling(
            params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out["allocation"] = (
            params.annual_vol_target / out["annual_vol"].clip(lower=0.01)
            * (0.20 + 0.80 * out["strength"])
        ).replace([np.inf, -np.inf], np.nan).fillna(0.01).clip(0.01, params.max_size)
        out["entry"] = (
            (out["contraction"] >= params.min_contraction)
            & (out["up_bias"] >= params.min_up_bias)
            & (out["return"] > 0)
            & out["annual_vol"].notna()
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.01, dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        allocation = indicators["allocation"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        pending = False
        active = False
        entry_price = 0.0
        held_bars = 0
        held_size = 0.01
        for i in range(n):
            # A decision on the preceding close fills at this bar's open.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_price = opens[i]
                    held_bars = 0
            if active:
                held_bars += 1
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or held_bars >= params.max_hold_bars:
                    active = False
                    # Exit at the next open, without reentering on this decision.
                    continue
                signal[i] = 1
                size[i] = held_size
            elif entry[i] and np.isfinite(allocation[i]) and allocation[i] > 0:
                held_size = float(np.clip(allocation[i], 0.01, params.max_size))
                signal[i] = 1
                size[i] = held_size
                pending = True
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100., 100., 110., 110., 110., 110.],
        "high": [101., 101., 112., 111., 111., 111.],
        "low": [99., 99., 109., 109., 109., 109.],
        "close": [100., 100., 111., 110., 110., 110.],
        "volume": [1000.] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [False, True, False, False, False, False],
        "allocation": [0.2, 0.4, 0.9, 0.9, 0.9, 0.9],
    }, index=index)
    target = strategy.generate_signals(
        data, indicators, None, Params(profit_target=0.005)).data
    assert target["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    timed = strategy.generate_signals(
        data, indicators, None, Params(profit_target=0.10)).data
    assert timed["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert timed.loc[index[2:4], "size"].eq(0.4).all()
    assert timed["size"].gt(0).all()
    derived = strategy.indicators(data, Params())
    assert not derived["entry"].any()
    assert derived["allocation"].gt(0).all()


if __name__ == "__main__":
    _self_check()
