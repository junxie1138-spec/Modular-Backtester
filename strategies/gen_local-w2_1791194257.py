from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 20
    atr_multiple: float = 2.5


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791194257"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return max(params.lookback + 1, 15)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.lookback < 2 or not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("lookback must be >= 2 and atr_multiple must be finite and positive")
        n = params.lookback
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        increments = out["log_return"].fillna(0.0)
        out["variance_increment"] = increments.square() if hasattr(increments, "square") else increments.pow(2)
        out["cumulative_return"] = increments.cumsum()
        out["cumulative_variance"] = out["variance_increment"].cumsum()
        out["cross_increment"] = (
            out["cumulative_variance"].shift(1).fillna(0.0) * increments
            - out["cumulative_return"].shift(1).fillna(0.0) * out["variance_increment"]
        )
        out["window_return"] = increments.rolling(n, min_periods=n).sum()
        out["window_variance"] = out["variance_increment"].rolling(n, min_periods=n).sum()
        out["window_variation"] = increments.abs().rolling(n, min_periods=n).sum()
        # Translate each window to its own origin before computing signed area.
        out["signature_area"] = 0.5 * (
            out["cross_increment"].rolling(n, min_periods=n).sum()
            - out["cumulative_variance"].shift(n) * out["window_return"]
            + out["cumulative_return"].shift(n) * out["window_variance"]
        )
        denominator = (out["window_variance"] * out["window_variation"]).replace(0.0, np.nan)
        out["area_score"] = (out["signature_area"] / denominator).replace([np.inf, -np.inf], np.nan)
        out["efficiency"] = out["window_return"] / out["window_variation"].replace(0.0, np.nan)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["ready"] = (
            out["log_return"].rolling(n, min_periods=n).count().eq(n)
            & np.isfinite(out["area_score"])
            & np.isfinite(out["efficiency"])
            & np.isfinite(out["atr"])
            & out["atr"].gt(0.0)
        )
        out["entry"] = (
            out["ready"] & out["area_score"].gt(0.04) & out["efficiency"].gt(0.25)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        if not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("atr_multiple must be finite and positive")
        count = len(data)
        target = np.zeros(count, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        holding = False
        armed = False
        entry_price = stop = peak_close = 0.0
        age = 0
        for i in range(count):
            # A preceding close decision becomes a position at this bar's open.
            if not holding and i > 0 and target[i - 1] == 1:
                if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                holding = True
                armed = False
                age = 0
                entry_price = opens[i]
                peak_close = entry_price
                stop = entry_price - params.atr_multiple * atr[i - 1]
            if holding:
                age += 1
                # Test the previously established stop before updating it.
                # A stop breach requests liquidation at the following open.
                breached = np.isfinite(lows[i]) and lows[i] <= stop
                if breached or age >= 20:
                    holding = False
                    continue
                if np.isfinite(closes[i]) and closes[i] > 0:
                    peak_close = max(peak_close, closes[i])
                    if closes[i] >= entry_price * 1.02:
                        armed = True
                    if armed:
                        stop = max(stop, entry_price)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, peak_close - params.atr_multiple * atr[i])
                target[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                target[i] = 1
        df = pd.DataFrame({"signal": target, "size": np.ones(count, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5, freq="B")
    data = pd.DataFrame({
        "open": [100.0, 100.0, 103.0, 103.0, 100.0],
        "high": [101.0, 104.0, 104.0, 104.0, 101.0],
        "low": [99.0, 99.0, 101.0, 100.0, 99.0],
        "close": [100.0, 103.0, 103.0, 101.0, 100.0],
        "volume": [1000.0] * 5,
    }, index=index)
    supplied = pd.DataFrame({"atr": [1.0, 1.0, 5.0, 1.0, 1.0], "entry": [True, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(data, supplied, None, SignatureParams()).data
    # ATR expansion cannot lower the armed stop; the breach exits one bar later.
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    flat = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=40, freq="B"))
    derived = GeneratedStrategy.indicators(flat, SignatureParams())
    assert derived.index.equals(flat.index)
    assert not derived["entry"].any()


if __name__ == "__main__":
    _self_check()
