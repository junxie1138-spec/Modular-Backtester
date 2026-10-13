from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeMemoryParams:
    correlation_window: int = 20
    percentile_window: int = 126
    tail_percentile: float = 0.80
    direction_window: int = 5
    profit_target: float = 0.03
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[RangeMemoryParams]):
    strategy_id = "gen_local-w2_1791332703"

    @classmethod
    def params_type(cls):
        return RangeMemoryParams

    def warmup_bars(self, params: RangeMemoryParams) -> int:
        return int(max(params.correlation_window + params.percentile_window + 2,
                       params.direction_window + 1))

    def indicators(self, data: pd.DataFrame, params: RangeMemoryParams) -> pd.DataFrame:
        if (params.correlation_window < 2 or params.percentile_window < 2
                or params.direction_window < 1 or params.holding_bars < 1
                or not 0.5 < params.tail_percentile < 1.0
                or not np.isfinite(params.profit_target) or params.profit_target <= 0):
            raise ValueError("Invalid range-memory parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["range"] = ((data["high"] - data["low"]).clip(lower=0)
                        / out["previous_close"]).replace([np.inf, -np.inf], np.nan)
        out["lagged_range"] = out["range"].shift(1)
        out["memory"] = out["range"].rolling(params.correlation_window).corr(
            out["lagged_range"]).replace([np.inf, -np.inf], np.nan)
        history = out["memory"].shift(1).rolling(params.percentile_window)
        out["upper"] = history.quantile(params.tail_percentile)
        out["lower"] = history.quantile(1.0 - params.tail_percentile)
        out["displacement"] = data["close"].diff(params.direction_window).replace(
            [np.inf, -np.inf], np.nan)
        out["direction"] = np.sign(out["displacement"].fillna(0)).astype(int)
        out["candidate"] = 0
        valid = out[["memory", "upper", "lower", "displacement"]].notna().all(axis=1)
        valid &= out["upper"] > out["lower"]
        out.loc[valid & (out["memory"] > out["upper"]), "candidate"] = out["direction"]
        out.loc[valid & (out["memory"] < out["lower"]), "candidate"] = -out["direction"]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeMemoryParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        memory = indicators["memory"].to_numpy(dtype=float)
        lower = indicators["lower"].to_numpy(dtype=float)
        upper = indicators["upper"].to_numpy(dtype=float)
        side, entry_bar, entry_price = 0, -1, np.nan
        armed = True
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if np.isfinite(memory[i]) and lower[i] <= memory[i] <= upper[i]:
                armed = True
            if side:
                # The previous close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        side = 0
                gain = side * (closes[i] / entry_price - 1.0) if side else np.nan
                # Evaluate targets at the close; exit at the following open.
                if (not side or gain >= params.profit_target
                        or i - entry_bar + 1 >= params.holding_bars):
                    side = 0
                    signals[i] = 0
                    continue
            elif i >= warmup and armed and candidates[i] != 0:
                if np.isfinite(closes[i]) and closes[i] > 0:
                    side = int(candidates[i])
                    entry_bar = i + 1
                    entry_price = np.nan
                    armed = False
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RangeMemoryParams(correlation_window=2, percentile_window=2,
                               direction_window=1, holding_bars=3)
    index = pd.date_range("2020-01-01", periods=14)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=6)
    for side in (-1, 1):
        for hit_target in (False, True):
            data = pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close", "volume")},
                                index=index)
            if hit_target:
                data.loc[index[8], "close"] = 100.0 * (1.0 + side * 0.04)
            ind = pd.DataFrame({"candidate": 0, "memory": 0.0,
                                "lower": -1.0, "upper": 1.0}, index=index)
            ind.loc[index[6], ["candidate", "memory"]] = [side, 2.0]
            result = strategy.generate_signals(data, ind, ctx, params).data
            exit_bar = 9 if hit_target else 10
            assert (result["signal"].iloc[:7] == 0).all()
            assert (result["signal"].iloc[7:exit_bar] == side).all()
            assert (result["signal"].iloc[exit_bar:] == 0).all()
            assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
