from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 5
    impulse_window: int = 10
    impulse_min: float = 0.008
    repayment_min: float = 0.35
    repayment_max: float = 0.75
    atr_bars: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 3.0
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791261953"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.roc_bars + params.impulse_window + 1, params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.roc_bars, params.impulse_window, params.atr_bars, params.max_hold) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (0 < params.repayment_min < params.repayment_max < 1):
            raise ValueError("Repayment bounds must lie strictly between zero and one.")
        if min(params.impulse_min, params.breakeven_pct, params.trail_atr) <= 0:
            raise ValueError("Impulse, activation and ATR multiplier must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["negative_impulse"] = (-out["acceleration"].clip(upper=0)).rolling(
            params.impulse_window, min_periods=params.impulse_window
        ).sum()
        out["positive_impulse"] = out["acceleration"].clip(lower=0).rolling(
            params.impulse_window, min_periods=params.impulse_window
        ).sum()
        out["long_repayment"] = out["positive_impulse"] / out["negative_impulse"].replace(0, np.nan)
        out["short_repayment"] = out["negative_impulse"] / out["positive_impulse"].replace(0, np.nan)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        ind = indicators
        valid = np.isfinite(ind).all(axis=1) & (ind["atr"] > 0)
        long_entry = (
            valid & (data["close"] > ind["ma200"]) & (ind["roc"] < 0)
            & (ind["acceleration"] > 0) & (ind["negative_impulse"] >= params.impulse_min)
            & ind["long_repayment"].between(params.repayment_min, params.repayment_max)
        )
        short_entry = (
            valid & (data["close"] < ind["ma200"]) & (ind["roc"] > 0)
            & (ind["acceleration"] < 0) & (ind["positive_impulse"] >= params.impulse_min)
            & ind["short_repayment"].between(params.repayment_min, params.repayment_max)
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = ind["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry = None
        stop = -np.inf
        armed = False
        held = 0
        for i in range(len(data)):
            if side:
                # The previous close's decision fills at this bar's open.
                if entry is None:
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                        entry = None
                        continue
                    stop = side * entry - params.trail_atr * atr[i - 1]
                    held = 0
                    armed = False
                held += 1
                if not valid.iloc[i] or not np.isfinite(closes[i]) or closes[i] <= 0:
                    side = 0
                    entry = None
                    continue
                signed_close = side * closes[i]
                # Signed-price stops only rise; short price stops consequently fall.
                if signed_close <= stop or held >= params.max_hold:
                    side = 0
                    entry = None
                    continue
                if side * (closes[i] / entry - 1.0) >= params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, side * entry, signed_close - params.trail_atr * atr[i])
            elif candidates[i]:
                side = int(candidates[i])
                entry = None
            raw[i] = side
        # ponytail: close-observed exits execute next open; intrabar execution needs a different contract.
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=7)
    for side in (1, -1):
        close = 100 + side * np.array([0, 0, 3, 2, 1, 0, 0], dtype=float)
        data = pd.DataFrame({"open": 100.0, "high": close + 1,
                             "low": close - 1, "close": close, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"ma200": 100 - side * 10, "roc": -side * 0.01,
                            "acceleration": [side * 0.001] + [0.0] * 6,
                            "negative_impulse": 0.01, "positive_impulse": 0.01,
                            "long_repayment": 0.5, "short_repayment": 0.5,
                            "atr": 0.5}, index=index)
        ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, side, side, side, 0, 0, 0]
        assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
