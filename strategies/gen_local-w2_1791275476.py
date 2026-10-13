from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuantileTransportParams:
    window: int = 20
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[QuantileTransportParams]):
    strategy_id = "gen_local-w2_1791275476"

    @classmethod
    def params_type(cls) -> type[QuantileTransportParams]:
        return QuantileTransportParams

    def warmup_bars(self, params: QuantileTransportParams) -> int:
        return int(max(2 * params.window + 1, 15))

    def indicators(self, data: pd.DataFrame, params: QuantileTransportParams) -> pd.DataFrame:
        if params.window < 5 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("window must be at least 5 and stop_atr must be positive")
        w = params.window
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lower"] = out["return"].rolling(w, min_periods=w).quantile(0.25)
        out["upper"] = out["return"].rolling(w, min_periods=w).quantile(0.75)
        out["baseline_lower"] = out["lower"].shift(w)
        out["baseline_upper"] = out["upper"].shift(w)
        out["volatility"] = out["return"].rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
        out["lower_move"] = (out["lower"] - out["baseline_lower"]) / out["volatility"]
        out["upper_move"] = (out["upper"] - out["baseline_upper"]) / out["volatility"]
        out["displacement"] = out["return"].rolling(3, min_periods=3).sum() / (out["volatility"] * np.sqrt(3.0))
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["regime"] = np.select([
            (out["lower_move"] > 0.1) & (out["upper_move"] > 0.1),
            (out["lower_move"] < -0.1) & (out["upper_move"] < -0.1),
            (out["lower_move"] < -0.1) & (out["upper_move"] > 0.1),
        ], [1, -1, 2], default=0).astype(int)
        out["entry"] = np.select([
            (out["regime"] == 1) & (out["displacement"] > 0.5),
            (out["regime"] == -1) & (out["displacement"] < -0.5),
            (out["regime"] == 2) & (out["displacement"] < -1.0),
            (out["regime"] == 2) & (out["displacement"] > 1.0),
        ], [1, -1, 1, -1], default=0).astype(int)
        out["spike"] = (out["return"].abs() > 2.5 * out["volatility"].shift(1)).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: QuantileTransportParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        atrs = indicators["atr"].to_numpy(dtype=float)
        spikes = indicators["spike"].fillna(False).to_numpy(dtype=bool)
        position = 0
        entry_price = np.nan
        entry_atr = np.nan
        held = 0
        blocked_until = -1
        for i in range(len(data)):
            if spikes[i]:
                blocked_until = max(blocked_until, i + 3)
            if position:
                if held == 0:
                    entry_price = opens[i]
                held += 1
                invalid = not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(closes[i])
                stopped = position * (closes[i] - entry_price) < -params.stop_atr * entry_atr
                if invalid or stopped or held >= 8:
                    position = 0
                    held = 0
                    blocked_until = max(blocked_until, i + 2)
            elif i > blocked_until and entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                position = int(entries[i])
                entry_atr = atrs[i]
                entry_price = np.nan
                held = 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = QuantileTransportParams()
    for direction in (1, -1):
        index = pd.date_range("2020-01-01", periods=12)
        prices = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
        derived = pd.DataFrame({"entry": 0, "atr": 1.0, "spike": False}, index=index)
        derived.iloc[0, derived.columns.get_loc("entry")] = direction
        result = strategy.generate_signals(prices, derived, None, params).data
        assert result["signal"].tolist() == [0] + [direction] * 8 + [0] * 3
        prices.iloc[2, prices.columns.get_loc("close")] = 100.0 - direction * 3.0
        result = strategy.generate_signals(prices, derived, None, params).data
        assert result["signal"].tolist() == [0, direction, direction] + [0] * 9
        assert (result["size"] > 0).all()
