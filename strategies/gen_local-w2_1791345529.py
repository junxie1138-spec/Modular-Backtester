from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    window: int = 16
    atr_multiple: float = 2.0


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791345529"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        return int(max(params.window, 15))

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.window < 4 or not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("window must be >= 4 and atr_multiple must be finite and positive")
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        values = log_close.to_numpy()
        upper_max = np.full(len(data), -np.inf)
        lower_max = np.full(len(data), -np.inf)
        upper_degree = np.zeros(len(data))
        lower_degree = np.zeros(len(data))
        for distance in range(1, params.window):
            slopes = (log_close.shift(distance).to_numpy() - values) / distance
            finite = np.isfinite(slopes)
            if distance > 1:
                upper_degree += finite & (slopes > upper_max)
                lower_degree += finite & (-slopes > lower_max)
            upper_max = np.maximum(upper_max, np.where(finite, slopes, -np.inf))
            lower_max = np.maximum(lower_max, np.where(finite, -slopes, -np.inf))
        valid_graph = log_close.rolling(params.window, min_periods=params.window).count().eq(params.window)
        denominator = float(params.window - 2)
        upper = pd.Series(upper_degree / denominator, index=data.index).where(valid_graph)
        lower = pd.Series(lower_degree / denominator, index=data.index).where(valid_graph)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        atr = true_range.rolling(14, min_periods=14).mean()
        displacement = log_close - log_close.shift(params.window - 1)
        return pd.DataFrame({
            "upper_degree": upper,
            "lower_degree": lower,
            "density": (upper + lower) / 2.0,
            "curvature": upper - lower,
            "displacement": displacement,
            "atr": atr,
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        if not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("atr_multiple must be finite and positive")
        ready = np.isfinite(indicators).all(axis=1) & indicators["atr"].gt(0)
        direction = np.sign(indicators["displacement"])
        curvature = indicators["curvature"]
        smooth = indicators["density"].ge(0.30)
        continuation = smooth & (direction * curvature).ge(0.15)
        reversal = ~smooth & (direction * curvature).le(-0.15)
        desired = pd.Series(np.where(continuation, direction,
                                     np.where(reversal, -direction, 0)), index=data.index)
        desired = desired.where(ready, 0).fillna(0).astype(int).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        valid = ready.to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        high_water = np.nan
        low_water = np.nan
        trailing_floor = -np.inf
        short_ceiling = np.inf
        for i in range(len(data)):
            price = closes[i]
            if not valid[i] or not np.isfinite(price) or price <= 0:
                position = 0
                continue
            if position != 0:
                # Entry decisions fill next bar; water marks begin on that bar's close.
                if i == entry_bar:
                    high_water = price
                    low_water = price
                else:
                    high_water = max(high_water, price)
                    low_water = min(low_water, price)
                trailing_floor = max(trailing_floor, high_water - params.atr_multiple * atrs[i])
                if position == -1:
                    short_ceiling = min(short_ceiling, low_water + params.atr_multiple * atrs[i])
                # The required rolling-high exit applies to both directions.
                # Shorts also have a mirrored protective trail against rising prices.
                stopped = price <= trailing_floor or (position == -1 and price >= short_ceiling)
                expired = i - entry_bar + 1 >= 8
                if stopped or expired:
                    position = 0
                signals[i] = position
                continue
            if desired[i] != 0:
                position = int(desired[i])
                entry_bar = i + 1
                high_water = np.nan
                low_water = np.nan
                trailing_floor = -np.inf
                short_ceiling = np.inf
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=32)
    close = np.full(32, 100.0)
    close[16:19] = [102.0, 101.0, 99.0]
    data = pd.DataFrame({"open": close, "high": close + 1.0,
                         "low": close - 1.0, "close": close,
                         "volume": 1000.0}, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert GeneratedStrategy.warmup_bars(params) >= 16
    controlled = pd.DataFrame({"upper_degree": 0.5, "lower_degree": 0.5,
                               "density": 0.5, "curvature": 0.0,
                               "displacement": 0.0, "atr": 1.0}, index=index)
    controlled.loc[index[15], ["curvature", "displacement"]] = [0.3, 0.1]
    controlled.loc[index[17:19], "atr"] = 3.0
    result = GeneratedStrategy.generate_signals(data, controlled, None, params).data
    assert result["signal"].iloc[15] == 0
    assert result["signal"].iloc[16] == 1
    assert result["signal"].iloc[18] == 1
    assert result["signal"].iloc[19] == 0
    controlled.loc[:, "curvature"] = 0.0
    controlled.loc[:, "displacement"] = 0.0
    controlled.loc[index[20], ["curvature", "displacement"]] = [-0.3, -0.1]
    result = GeneratedStrategy.generate_signals(data, controlled, None, params).data
    assert result["signal"].iloc[21] == -1
    assert result["signal"].iloc[28] == -1
    assert result["signal"].iloc[29] == 0
    assert result["signal"].isin([-1, 0, 1]).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
