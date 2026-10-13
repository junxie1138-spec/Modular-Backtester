from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 6
    area_threshold: float = 0.08
    efficiency_threshold: float = 0.55
    pullback: float = 0.008
    profit_target: float = 0.035
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791264625"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        if not 2 <= params.lookback <= 8:
            raise ValueError("lookback must be between 2 and 8")
        if not 0.0 < params.area_threshold < 0.5:
            raise ValueError("area_threshold must be between 0 and 0.5")
        if not 0.0 < params.efficiency_threshold < 1.0:
            raise ValueError("efficiency_threshold must be between 0 and 1")
        if params.pullback <= 0.0 or params.profit_target <= 0.0:
            raise ValueError("pullback and profit_target must be positive")
        if params.max_hold < 1:
            raise ValueError("max_hold must be positive")
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        self.warmup_bars(params)
        n = params.lookback
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        out["return"] = (close / close.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["range"] = ((data["high"] - data["low"]) / close).replace([np.inf, -np.inf], np.nan)
        out["range_change"] = out["range"].diff()
        x = out["return"]
        y = out["range_change"]
        out["displacement"] = x.rolling(n, min_periods=n).sum()
        out["travel"] = x.abs().rolling(n, min_periods=n).sum()
        out["range_travel"] = y.abs().rolling(n, min_periods=n).sum()
        out["efficiency"] = out["displacement"] / out["travel"].replace(0.0, np.nan)
        area = pd.Series(0.0, index=data.index)
        # Older increments precede newer increments in the signed area.
        for older in range(1, n):
            for newer in range(older):
                area += 0.5 * (x.shift(older) * y.shift(newer) - y.shift(older) * x.shift(newer))
        denominator = (out["travel"] * out["range_travel"]).replace(0.0, np.nan)
        out["signature_area"] = (area / denominator).replace([np.inf, -np.inf], np.nan)
        ready = out[["efficiency", "signature_area", "displacement", "return"]].notna().all(axis=1)
        directional = out["efficiency"].abs() >= params.efficiency_threshold
        out["directional_regime"] = directional.astype(int)
        continuation = directional & (out["efficiency"] > 0.0) & (x > 0.0)
        rebound = (~directional) & (out["displacement"] <= -params.pullback) & (x > 0.0)
        out["entry"] = (ready & (out["signature_area"] > params.area_threshold) & (continuation | rebound)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        holding = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if holding:
                # Yesterday's entry decision fills at today's open.
                if entry_bar < 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        holding = False
                        continue
                    entry_bar = i
                    entry_price = opens[i]
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.max_hold
                if target_hit or time_up:
                    holding = False
                    entry_bar = -1
                    entry_price = np.nan
                    continue
                signal[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                holding = True
                entry_bar = -1
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    forced = pd.DataFrame({"entry": [True] + [False] * 23}, index=index)
    timed = strategy.generate_signals(data, forced, None, params).data
    assert timed["signal"].tolist() == [0] + [1] * 18 + [0] * 5
    data.loc[index[1], "close"] = 104.0
    targeted = strategy.generate_signals(data, forced, None, params).data
    assert targeted["signal"].tolist() == [0, 1] + [0] * 22
    assert targeted["size"].gt(0.0).all()
    assert strategy.warmup_bars(params) <= 10
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["entry"].iloc[:params.lookback].any()


if __name__ == "__main__":
    _self_check()
