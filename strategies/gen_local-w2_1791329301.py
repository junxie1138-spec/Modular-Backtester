from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeEchoParams:
    lookback: int = 20
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[RangeEchoParams]):
    strategy_id = "gen_local-w2_1791329301"

    @classmethod
    def params_type(cls) -> type[RangeEchoParams]:
        return RangeEchoParams

    @staticmethod
    def warmup_bars(params: RangeEchoParams) -> int:
        return max(14, int(params.lookback) + 3)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeEchoParams) -> pd.DataFrame:
        if params.lookback < 3 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("lookback must be >= 3 and stop_atr must be positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        out["range_fraction"] = (data["high"] - data["low"]) / previous_close
        out["range_change"] = out["range_fraction"].diff()
        out["lagged_range_change"] = out["range_change"].shift(1)
        out["range_autocorrelation"] = out["range_change"].rolling(
            params.lookback, min_periods=params.lookback
        ).corr(out["lagged_range_change"])
        out["range_forecast"] = out["range_autocorrelation"] * out["range_change"]
        out["candle_direction"] = np.sign(data["close"] - data["open"])
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1) & out["atr"].gt(0)
        qualified = ready & out["range_autocorrelation"].abs().ge(0.2)
        out["entry"] = (
            np.sign(out["range_forecast"]) * out["candle_direction"]
        ).where(qualified, 0).fillna(0).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeEchoParams,
    ) -> SignalFrame:
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive")
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        held = 0
        highest_close = lowest_close = np.nan
        high_stop = -np.inf
        short_stop = np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if position:
                held += 1
                if not valid:
                    position = 0
                else:
                    # The first held bar is the actual next-bar entry session.
                    if held == 1:
                        highest_close = lowest_close = close
                        high_stop = close - params.stop_atr * atr
                        short_stop = close + params.stop_atr * atr
                    else:
                        highest_close = max(highest_close, close)
                        lowest_close = min(lowest_close, close)
                        high_stop = max(high_stop, highest_close - params.stop_atr * atr)
                        short_stop = min(short_stop, lowest_close + params.stop_atr * atr)
                    # Required high-water exit applies to both directions.
                    # Shorts additionally have a mirrored adverse-price trail.
                    if close < high_stop or (position == -1 and close > short_stop) or held >= 4:
                        position = 0
                raw[i] = position
                continue
            if valid and entries[i] in (-1, 1):
                position = int(entries[i])
                held = 0
                highest_close = lowest_close = np.nan
                high_stop = -np.inf
                short_stop = np.inf
                raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    params = RangeEchoParams()
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 99.0,
        "close": [100.0, 103.0, 102.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    features = pd.DataFrame({"atr": 1.0, "entry": [1, 0, 0, 0, 0, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    # Rising ATR cannot loosen an existing stop.
    features.loc[index[3], "atr"] = 10.0
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    data["close"] = 100.0
    features["atr"] = 1.0
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    features["entry"] = [-1, 0, 0, 0, 0, 0, 0]
    data.loc[index[2], "close"] = 103.0
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, 0, 0, 0]
    features["atr"] = np.nan
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].eq(0).all()
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= params.lookback + 3


if __name__ == "__main__":
    _self_check()
