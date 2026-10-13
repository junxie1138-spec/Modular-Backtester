from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_window: int = 12
    reference_window: int = 40
    atr_window: int = 14
    change_threshold: float = 1.5
    stop_atr: float = 2.5
    hold_bars: int = 18

    def __post_init__(self):
        if self.path_window < 2 or self.reference_window < 2:
            raise ValueError("Path and reference windows must be at least two.")
        if self.atr_window < 1 or self.hold_bars < 1:
            raise ValueError("ATR and holding windows must be positive.")
        if not np.isfinite(self.change_threshold) or self.change_threshold <= 0:
            raise ValueError("Change threshold must be finite and positive.")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("Stop multiplier must be finite and positive.")


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790977149"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(200, params.path_window + params.reference_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_price"] = np.log(close.where(close > 0))
        out["return"] = out["log_price"].diff()
        out["bar_number"] = np.arange(len(data), dtype=float)
        out["weighted_return"] = out["bar_number"] * out["return"]
        length = params.path_window
        out["path_return"] = out["return"].rolling(length).sum()
        out["weighted_sum"] = out["weighted_return"].rolling(length).sum()
        # Half the antisymmetric level-two signature of (time, log price).
        out["signature_area"] = (
            out["weighted_sum"]
            - (out["bar_number"] - (length - 1) / 2.0) * out["path_return"]
        )
        out["path_volatility"] = out["return"].rolling(length).std(ddof=0)
        scale = out["path_volatility"].replace(0.0, np.nan) * length ** 1.5
        out["normalized_area"] = out["signature_area"] / scale
        history = out["normalized_area"].shift(1)
        out["area_mean"] = history.rolling(params.reference_window).mean()
        out["area_std"] = history.rolling(params.reference_window).std(ddof=0)
        out["change_score"] = (
            (out["normalized_area"] - out["area_mean"])
            / out["area_std"].where(out["area_std"] > 1e-12)
        )
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ma200"] = close.rolling(200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SignatureParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        scores = indicators["change_score"].to_numpy(dtype=float)
        areas = indicators["normalized_area"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        armed = True
        entry_bar = -1
        stop_price = np.nan
        for i in range(len(data)):
            if np.isfinite(scores[i]) and scores[i] <= 0.0:
                armed = True
            if active:
                if i == entry_bar:
                    # The preceding close's decision fills at this bar's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        continue
                    stop_price = opens[i] - params.stop_atr * atr[i - 1]
                expired = i - entry_bar + 1 >= params.hold_bars
                stopped = np.isfinite(closes[i]) and closes[i] < stop_price
                if expired or stopped:
                    active = False
                else:
                    signals[i] = 1
                continue
            eligible = (
                armed and np.isfinite(scores[i]) and np.isfinite(areas[i])
                and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(regime[i]) and np.isfinite(closes[i])
                and closes[i] > regime[i] and areas[i] > 0
                and scores[i] >= params.change_threshold
            )
            if eligible:
                active = True
                armed = False
                entry_bar = i + 1
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    data = pd.DataFrame(
        {"open": 110.0, "high": 112.0, "low": 106.0,
         "close": 110.0, "volume": 1000.0}, index=index,
    )
    ind = pd.DataFrame(
        {"change_score": 0.0, "normalized_area": 1.0,
         "atr": 1.0, "ma200": 90.0}, index=index,
    )
    ind.iloc[0, 0] = 2.0
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0] + [1] * 18 + [0] * 11
    data.loc[index[0], "close"] = 100.0
    data.loc[index[2], "close"] = 107.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1] + [0] * 27
    ind["ma200"] = 200.0
    assert not strategy.generate_signals(data, ind, ctx, params).data["signal"].any()
    index = pd.date_range("2020-01-01", periods=320, freq="B")
    x = np.arange(320, dtype=float)
    close = 100.0 * np.exp(0.0002 * x + 0.000001 * x * x + 0.003 * np.sin(x))
    data = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0,
         "close": close, "volume": 1000.0}, index=index,
    )
    full = strategy.indicators(data, params)
    assert full.iloc[:250].equals(strategy.indicators(data.iloc[:250], params))
    result = strategy.generate_signals(data, full, ctx, params).data
    assert result.index.equals(data.index)
    assert result["signal"].isin([0, 1]).all() and (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
