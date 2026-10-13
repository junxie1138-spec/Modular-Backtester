from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 8
    area_threshold: float = 0.20
    efficiency_min: float = 0.35
    atr_bars: int = 14
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791292275"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return max(200, params.lookback + 2, params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if not (params.lookback >= 2 and params.atr_bars >= 1
                and 3 <= params.hold_bars <= 5
                and 0 < params.area_threshold < 1
                and 0 <= params.efficiency_min <= 1
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["log_return"] = np.log(close.where(close > 0)).diff()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["ma200"] = close.rolling(200).mean()
        n = params.lookback
        r = out["log_return"]
        out["path_length"] = r.abs().rolling(n).sum()
        out["displacement"] = r.rolling(n).sum()
        # Antisymmetric level-two signature of (elapsed time, log price).
        # Older returns have negative weights; later returns have positive weights.
        area = pd.Series(0.0, index=data.index)
        for lag in range(n):
            area = area + 0.5 * (n - 1 - 2 * lag) * r.shift(lag)
        out["signature_area"] = area
        denominator = out["path_length"].where(out["path_length"] > 0)
        out["area_score"] = (2 * area / (n * denominator)).replace(
            [np.inf, -np.inf], np.nan
        )
        out["efficiency"] = (out["displacement"] / denominator).replace(
            [np.inf, -np.inf], np.nan
        )
        score = out["area_score"]
        out["entry"] = (
            score.ge(params.area_threshold)
            & score.shift(1).lt(params.area_threshold)
            & out["efficiency"].ge(params.efficiency_min)
            & close.gt(out["ma200"])
            & out["atr"].gt(0)
            & np.isfinite(out["atr"])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SignatureParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if pending:
                # Only the current open and the previous decision's ATR are used.
                active = bool(np.isfinite(opens[i]) and opens[i] > 0)
                stop = opens[i] - params.stop_atr * atr[i - 1]
                entry_bar = i
                pending = False
            if active:
                if (not np.isfinite(closes[i]) or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    active = False
                raw[i] = int(active)
            elif entries[i] and i + 1 < len(data):
                raw[i] = 1
                pending = True
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=9, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 111.0, "low": 94.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    ind = pd.DataFrame(
        {"atr": 2.0, "entry": [True] + [False] * 8}, index=index
    )
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    signals = strategy.generate_signals(data, ind, ctx, SignatureParams()).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0]
    data.loc[index[2], "close"] = 110.0
    data.loc[index[4], "close"] = 96.0
    signals = strategy.generate_signals(
        data, ind, ctx, SignatureParams(hold_bars=5)
    ).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0]
    data.loc[index[1], "open"] = 105.0
    signals = strategy.generate_signals(data, ind, ctx, SignatureParams()).data
    assert signals["signal"].tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
