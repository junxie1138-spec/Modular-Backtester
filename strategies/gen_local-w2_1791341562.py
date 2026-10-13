from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    shock_threshold: float = 1.5
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local-w2_1791341562"

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return 90

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if not np.isfinite(params.shock_threshold) or params.shock_threshold <= 0:
            raise ValueError("shock_threshold must be finite and positive")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        close = data["close"].astype(float)
        log_price = np.log(close.where(close > 0))
        returns = log_price.diff()
        slope = returns.rolling(10, min_periods=10).mean()
        expected_slope = slope.shift(1).rolling(5, min_periods=5).mean()
        innovation = slope - expected_slope
        scale = innovation.shift(1).rolling(63, min_periods=63).std()
        z = innovation.div(scale.where(scale > 0)).replace([np.inf, -np.inf], np.nan)
        weekly_corr = innovation.rolling(63, min_periods=63).corr(innovation.shift(5))
        prior_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior_close).abs(),
            (data["low"] - prior_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        quiet_max = z.shift(1).abs().rolling(4, min_periods=4).max()
        entry = (
            (z.shift(5) > params.shock_threshold)
            & (quiet_max < params.shock_threshold)
            & (z > 0)
            & (z < params.shock_threshold)
            & (slope > 0)
            & (weekly_corr > 0.15)
            & (atr > 0)
        ).fillna(False)
        out = pd.DataFrame({
            "log_price": log_price,
            "returns": returns,
            "slope": slope,
            "expected_slope": expected_slope,
            "innovation": innovation,
            "innovation_scale": scale,
            "innovation_z": z,
            "weekly_corr": weekly_corr,
            "true_range": true_range,
            "atr": atr,
            "quiet_max": quiet_max,
            "entry": entry,
        }, index=data.index)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EchoParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        fixed_stop = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Entry fills at this bar's open; ATR was known at the decision close.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        continue
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
                if (
                    not np.isfinite(closes[i])
                    or closes[i] < fixed_stop
                    or i - entry_bar + 1 >= 18
                ):
                    active = False
                    continue
                signal[i] = 1
            elif i >= self.warmup_bars(params) - 1 and entries[i]:
                active = True
                entry_bar = i + 1
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=115, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = EchoParams()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    derived["entry"] = False
    derived.loc[index[90], "entry"] = True
    derived["atr"] = 2.0
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].iloc[90] == 0
    assert result["signal"].iloc[91:109].eq(1).all()
    assert result["signal"].iloc[109] == 0
    data.loc[index[92], "close"] = 96.0
    data.loc[index[93], "close"] = 94.0
    derived.loc[index[92], "atr"] = 100.0
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].iloc[93] == 1
    assert result["signal"].iloc[94] == 0
    assert set(result["signal"].unique()) <= {0, 1}
    assert result["size"].gt(0).all()
