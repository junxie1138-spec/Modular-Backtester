from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 12
    correlation_window: int = 40
    area_threshold: float = 0.25
    correlation_threshold: float = 0.10
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790870458"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return max(200, params.signature_window + 1, params.correlation_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.signature_window < 2 or params.correlation_window < 3:
            raise ValueError("Indicator windows must be at least 2 and 3 bars respectively")
        n = params.signature_window
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        opening = data["open"].where(lambda s: s > 0)
        gap = (opening / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        body = (data["close"] / opening - 1.0).replace([np.inf, -np.inf], np.nan)
        valid = gap.notna() & body.notna()
        x = gap.where(valid, 0.0)
        y = body.where(valid, 0.0)
        cumulative_x = x.cumsum()
        cumulative_y = y.cumsum()
        before_x = cumulative_x.shift(1).fillna(0.0)
        before_y = cumulative_y.shift(1).fillna(0.0)
        start_x = cumulative_x.shift(n).fillna(0.0)
        start_y = cumulative_y.shift(n).fillna(0.0)
        sum_x = x.rolling(n, min_periods=n).sum()
        sum_y = y.rolling(n, min_periods=n).sum()
        cross = (before_x * y - before_y * x).rolling(n, min_periods=n).sum()
        signed_area = 0.5 * (cross - start_x * sum_y + start_y * sum_x)
        energy = (x.square() + y.square()) if hasattr(x, "square") else (x * x + y * y)
        energy = energy.rolling(n, min_periods=n).sum()
        complete = valid.astype(int).rolling(n, min_periods=n).sum().eq(n)
        normalized_area = (signed_area / energy.where(energy > 0)).where(complete)
        correlation = body.rolling(params.correlation_window, min_periods=params.correlation_window).corr(body.shift(1))
        return pd.DataFrame({
            "overnight_return": gap,
            "intraday_return": body,
            "signature_area": normalized_area.replace([np.inf, -np.inf], np.nan),
            "intraday_autocorrelation": correlation.replace([np.inf, -np.inf], np.nan),
            "ma200": data["close"].rolling(200, min_periods=200).mean(),
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        if params.max_hold_bars < 1 or params.profit_target <= 0:
            raise ValueError("Holding period and profit target must be positive")
        eligible = (
            indicators["signature_area"].gt(params.area_threshold)
            & indicators["intraday_autocorrelation"].gt(params.correlation_threshold)
            & data["close"].gt(indicators["ma200"])
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        entry_price = 0.0
        entry_bar = 0
        warmup = GeneratedStrategy.warmup_bars(params)
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_price = opens[i]
                    entry_bar = i
            if active:
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_expired:
                    active = False
                else:
                    signal[i] = 1
            elif i >= warmup - 1 and eligible[i]:
                signal[i] = 1
                pending_entry = True
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=240, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    computed = GeneratedStrategy.indicators(data, params)
    assert computed.index.equals(data.index)
    assert computed["signature_area"].isna().all()
    features = pd.DataFrame({"signature_area": 1.0, "intraday_autocorrelation": 0.5, "ma200": 90.0}, index=index)
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].iloc[:200].eq(0).all()
    assert result["signal"].iloc[200:218].eq(1).all()
    assert result["signal"].iloc[218] == 0
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
    data.loc[index[205], "close"] = 105.0
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].iloc[205] == 1 and result["signal"].iloc[206] == 0
