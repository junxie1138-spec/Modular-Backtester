from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    breakeven_pct: float = 0.004
    trail_atr: float = 0.9


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791268024"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 32

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["volume_max"] = data["volume"].shift(1).rolling(20, min_periods=20).max()
        out["volume_median"] = data["volume"].shift(1).rolling(20, min_periods=20).median()
        out["spike"] = (data["volume"] > 1.25 * out["volume_max"]) & (out["volume_max"] > 0)
        bar = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        out["anchor_age"] = bar - bar.where(out["spike"]).ffill(limit=10)
        out["anchor_high"] = data["high"].where(out["spike"]).ffill(limit=10)
        out["anchor_low"] = data["low"].where(out["spike"]).ffill(limit=10)
        out["anchor_width"] = (out["anchor_high"] - out["anchor_low"]).replace(0.0, np.nan)
        out["position"] = (data["close"] - out["anchor_low"]) / out["anchor_width"]
        out["previous_position"] = (previous - out["anchor_low"]) / out["anchor_width"]
        out["volume_ratio"] = data["volume"] / out["volume_median"].replace(0.0, np.nan)
        out["body"] = data["close"] - data["open"]
        ready = (
            out["anchor_age"].between(2, 10)
            & out["position"].between(0, 1)
            & (out["volume_ratio"] >= 1.2)
            & (out["atr"] > 0)
        )
        up = ready & (out["previous_position"] <= 0.5) & (out["position"] > 0.5) & (out["body"] > 0)
        down = ready & (out["previous_position"] >= 0.5) & (out["position"] < 0.5) & (out["body"] < 0)
        out["candidate"] = np.select([up, down], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        if not (np.isfinite(params.breakeven_pct) and params.breakeven_pct > 0
                and np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("Exit parameters must be finite and positive.")
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        side = 0
        pending = False
        age = 0
        entry = best = stop = 0.0
        armed = False
        for i in range(len(data)):
            if pending:
                # Previous close's decision fills at this bar's open.
                entry = opens[i]
                if not (np.isfinite(entry) and entry > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0):
                    side = 0
                else:
                    best = side * entry
                    stop = best - params.trail_atr * atrs[i - 1]
                    age = 0
                    armed = False
                pending = False
            if side:
                age += 1
                price = side * closes[i]
                if not (np.isfinite(price) and np.isfinite(atrs[i]) and atrs[i] > 0):
                    side = 0
                elif price <= stop:
                    side = 0
                else:
                    best = max(best, price)
                    if best - side * entry >= entry * params.breakeven_pct:
                        armed = True
                    if armed:
                        # Signed stop rises monotonically; a short's price stop falls.
                        stop = max(stop, side * entry, best - params.trail_atr * atrs[i])
                    # Stops are close-observed; liquidation occurs next open.
                    if price <= stop or age >= 2:
                        side = 0
                raw[i] = side
                continue
            if i + 1 >= self.warmup_bars(params) and candidates[i] in (-1, 1):
                if np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0:
                    side = int(candidates[i])
                    pending = True
                    raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams(breakeven_pct=0.005, trail_atr=0.5)
    index = pd.date_range("2020-01-01", periods=36)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=32)
    for direction in (1, -1):
        data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0,
                             "close": 100.0, "volume": 100.0}, index=index)
        data.iloc[32, data.columns.get_loc("close")] = 100.0 + direction
        ind = strategy.indicators(data, params)
        assert ind.index.equals(index) and not ind["candidate"].iloc[:20].any()
        ind["atr"] = 1.0
        ind["candidate"] = 0
        ind.iloc[31, ind.columns.get_loc("candidate")] = direction
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].iloc[31:35].tolist() == [0, direction, direction, 0]
        assert (result["size"] > 0).all()
        prefix = strategy.generate_signals(data.iloc[:33], ind.iloc[:33], ctx, params).data
        assert result.iloc[:33].equals(prefix)
        data.iloc[32, data.columns.get_loc("close")] = 100.0 - direction
        stopped = strategy.generate_signals(data, ind, ctx, params).data
        assert stopped["signal"].iloc[31:35].tolist() == [0, direction, 0, 0]


if __name__ == "__main__":
    _self_check()
