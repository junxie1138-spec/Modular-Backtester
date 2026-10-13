from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_window: int = 12
    corr_window: int = 40
    area_threshold: float = 0.12
    corr_threshold: float = 0.08
    atr_window: int = 14
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791311134"

    @classmethod
    def params_type(cls):
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return int(max(params.area_window + 1, params.corr_window + 2, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["lagged_return"] = out["return"].shift(1)
        length = params.area_window
        weights = 2.0 * np.arange(length) - length + 1.0
        if len(data):
            weighted = np.convolve(out["return"].fillna(0).to_numpy(), weights[::-1], mode="full")[:len(data)]
        else:
            weighted = np.empty(0, dtype=float)
        out["return_sum"] = out["return"].rolling(length).sum()
        out["return_variation"] = out["return"].abs().rolling(length).sum()
        # Antisymmetric level-two signature of the (time, log-price) path.
        out["area"] = pd.Series(weighted, index=data.index) / (length * out["return_variation"].replace(0, np.nan))
        out["autocorr"] = out["return"].rolling(params.corr_window).corr(out["lagged_return"]).clip(-1, 1)
        out["continuation"] = out["autocorr"] * out["return_sum"]
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        valid = indicators[["area", "autocorr", "continuation", "atr"]].notna().all(axis=1)
        valid &= indicators["atr"] > 0
        persistent = indicators["autocorr"] > params.corr_threshold
        buy = valid & persistent & (indicators["area"] > params.area_threshold) & (indicators["continuation"] > 0)
        sell = valid & persistent & (indicators["area"] < -params.area_threshold) & (indicators["continuation"] < 0)
        candidate = np.select([buy, sell], [1, -1], default=0)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry = best = stop = 0.0
        entered = 0
        armed = False
        for i, (opening, high, low, closing) in enumerate(prices):
            if not np.isfinite([opening, high, low, closing]).all() or min(opening, high, low, closing) <= 0:
                position = 0
                continue
            if position == 0 and i > 0 and raw[i - 1] != 0:
                # The previous close's decision fills at this bar's open.
                position = int(raw[i - 1])
                entry = position * opening
                best = entry
                stop = entry - params.trail_atr * atr[i - 1]
                entered = i
                armed = False
            if position:
                adverse = position * (low if position == 1 else high)
                favorable = position * (high if position == 1 else low)
                signed_close = position * closing
                # Test the previously established stop before updating it.
                exit_now = adverse <= stop or i - entered + 1 >= params.hold_bars
                if not exit_now:
                    best = max(best, favorable)
                    armed = armed or best - entry >= abs(entry) * params.breakeven_pct
                    if armed and np.isfinite(atr[i]) and atr[i] > 0:
                        # Signed prices make the short-side ratchet upward too.
                        stop = max(stop, entry, best - params.trail_atr * atr[i])
                        exit_now = signed_close <= stop
                if exit_now:
                    position = 0
                else:
                    raw[i] = position
            else:
                raw[i] = int(candidate[i])
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = SignatureParams(trail_atr=2.0, breakeven_pct=0.02)
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100., 100., 103., 104., 100.],
        "high": [100., 104., 105., 104., 100.],
        "low": [99., 100., 103., 99., 99.],
        "close": [100., 103., 104., 100., 100.],
        "volume": 1000.,
    }, index=index)
    for direction in (1, -1):
        bars = data.copy()
        if direction == -1:
            bars["open"] = 200. - data["open"]
            bars["close"] = 200. - data["close"]
            bars["high"] = 200. - data["low"]
            bars["low"] = 200. - data["high"]
        features = pd.DataFrame({
            "area": [direction * 0.5, 0., 0., 0., 0.],
            "autocorr": 0.5,
            "continuation": [direction * 0.01, 0., 0., 0., 0.],
            "atr": 1.,
        }, index=index)
        result = GeneratedStrategy.generate_signals(bars, features, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0]
        features.loc[index[0], "continuation"] *= -1
        assert not GeneratedStrategy.generate_signals(bars, features, None, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
