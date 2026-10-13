from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 5
    history_bars: int = 252
    horizon_bars: int = 4
    analog_quantile: float = 0.10
    entry_quantile: float = 0.65


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1790961521"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(2 * params.history_bars + 2 * params.path_bars
                   + 2 * params.horizon_bars + 2)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        p = params
        if p.path_bars < 2 or p.history_bars < 20 or not 3 <= p.horizon_bars <= 5:
            raise ValueError("Require path_bars >= 2, history_bars >= 20, and horizon_bars in [3, 5].")
        if not 0 < p.analog_quantile < 1 or not 0 < p.entry_quantile < 1:
            raise ValueError("Quantiles must be strictly between zero and one.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["gap"] = np.log(opening / close.shift(1))
        out["intraday"] = np.log(close / opening)
        gap_scale = out["gap"].abs().rolling(p.path_bars).sum()
        day_scale = out["intraday"].abs().rolling(p.path_bars).sum()
        out["gap_scale"] = gap_scale
        out["intraday_scale"] = day_scale
        x = pd.Series(0.0, index=data.index)
        y = pd.Series(0.0, index=data.index)
        area = pd.Series(0.0, index=data.index)
        for lag in range(p.path_bars - 1, -1, -1):
            dx = out["gap"].shift(lag)
            dy = out["intraday"].shift(lag)
            area = area + 0.5 * (x * dy - y * dx)
            x = x + dx
            y = y + dy
        out["gap_balance"] = x / gap_scale.replace(0.0, np.nan)
        out["intraday_balance"] = y / day_scale.replace(0.0, np.nan)
        out["signature_area"] = area / (gap_scale * day_scale).replace(0.0, np.nan)
        out["matured_return"] = np.log(opening / opening.shift(p.horizon_bars))
        features = out[["signature_area", "gap_balance", "intraday_balance"]].to_numpy()
        outcomes = out["matured_return"].to_numpy()
        forecast = np.full(len(data), np.nan)
        # ponytail: O(bars * history_bars) scan; index signatures if histories become large.
        for t in range(len(data)):
            end = t - p.horizon_bars - 1 - p.path_bars
            start = end - p.history_bars + 1
            if start < p.path_bars or not np.isfinite(features[t]).all():
                continue
            candidates = features[start:end + 1]
            realized = outcomes[start + p.horizon_bars + 1:end + p.horizon_bars + 2]
            valid = np.isfinite(candidates).all(axis=1) & np.isfinite(realized)
            if valid.sum() < p.history_bars:
                continue
            difference = candidates[valid] - features[t]
            distance = 4.0 * difference[:, 0] ** 2 + (difference[:, 1:] ** 2).sum(axis=1)
            selected = distance <= np.quantile(distance, p.analog_quantile)
            forecast[t] = realized[valid][selected].mean()
        out["analog_forecast"] = forecast
        out["forecast"] = out["analog_forecast"].rolling(p.horizon_bars).mean()
        out["entry_threshold"] = out["forecast"].shift(1).rolling(p.history_bars).quantile(p.entry_quantile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SignatureParams,
    ) -> SignalFrame:
        active = (
            indicators["forecast"].notna()
            & indicators["entry_threshold"].notna()
            & (indicators["forecast"] > indicators["entry_threshold"])
            & (indicators["forecast"] > 0.0)
        )
        df = pd.DataFrame(index=data.index)
        # A false entry condition reverses the long thesis and exits the position.
        df["signal"] = active.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    index = pd.date_range("2015-01-01", periods=1000, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(index))))
    opening = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0.0, 0.004, len(index)))
    data = pd.DataFrame({
        "open": opening, "high": np.maximum(opening, close) * 1.01,
        "low": np.minimum(opening, close) * 0.99,
        "close": close, "volume": 1000000.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = SignatureParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    prefix = data.iloc[:800]
    prefix_indicators = strategy.indicators(prefix, params)
    pd.testing.assert_frame_equal(indicators.iloc[:800], prefix_indicators)
    pd.testing.assert_frame_equal(
        signals.iloc[:800], strategy.generate_signals(prefix, prefix_indicators, ctx, params).data,
    )
    expected = ((indicators["forecast"] > indicators["entry_threshold"])
                & (indicators["forecast"] > 0)).astype(int).shift(1).fillna(0).astype(int)
    pd.testing.assert_series_equal(signals["signal"], expected, check_names=False)
    assert signals["signal"].isin([0, 1]).all() and signals["signal"].sum() > 0
    assert (signals["size"] > 0).all() and signals["signal"].iloc[0] == 0
    assert indicators["entry_threshold"].iloc[strategy.warmup_bars(params):].notna().all()
