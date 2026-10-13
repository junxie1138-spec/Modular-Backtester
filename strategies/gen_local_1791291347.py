from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    memory: int = 80
    vol_window: int = 20
    atr_window: int = 14
    min_samples: int = 10
    shock_sigma: float = 1.0
    correlation_gap: float = 0.25
    forecast_threshold: float = 0.10
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791291347"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.memory + params.vol_window + 3, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lag_return"] = out["return"].shift(1)
        out["volatility"] = out["return"].rolling(params.vol_window).std().shift(1)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["training_shock"] = out["lag_return"].abs().ge(
            params.shock_sigma * out["volatility"].shift(1)
        )
        training_valid = (
            out["volatility"].shift(1).gt(0)
            & out["return"].notna()
            & out["lag_return"].notna()
        )
        for name, mask in (
            ("shock", training_valid & out["training_shock"]),
            ("quiet", training_valid & ~out["training_shock"]),
        ):
            x = out["lag_return"].where(mask)
            y = out["return"].where(mask)
            rx = x.rolling(params.memory, min_periods=params.min_samples)
            ry = y.rolling(params.memory, min_periods=params.min_samples)
            out[name + "_correlation"] = rx.corr(y)
            out[name + "_slope"] = rx.cov(y).div(rx.var().replace(0.0, np.nan))
            out[name + "_forecast"] = (
                ry.mean() + out[name + "_slope"] * (out["return"] - rx.mean())
            )
        out["current_shock"] = out["return"].abs().ge(params.shock_sigma * out["volatility"])
        out["forecast"] = out["shock_forecast"].where(
            out["current_shock"], out["quiet_forecast"]
        )
        out["score"] = out["forecast"].div(out["volatility"].replace(0.0, np.nan))
        out["regime"] = (
            out["shock_correlation"].gt(0.0)
            & out["quiet_correlation"].lt(0.0)
            & (out["shock_correlation"] - out["quiet_correlation"]).ge(params.correlation_gap)
        )
        out["candidate"] = np.select([
            out["regime"] & out["score"].ge(params.forecast_threshold),
            out["regime"] & out["score"].le(-params.forecast_threshold),
        ], [1, -1], default=0).astype(int)
        out.iloc[:self.warmup_bars(params), out.columns.get_loc("candidate")] = 0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        candidates = indicators["candidate"].fillna(0).astype(int)
        confirmed = candidates.where(candidates.eq(candidates.shift(1)), 0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # The previous close requested entry; this bar's open anchors the fixed stop.
                    stop = opens[i] - position * params.stop_atr * atr[i - 1]
                invalid = not np.isfinite(closes[i]) or not np.isfinite(stop)
                stopped = closes[i] < stop if position == 1 else closes[i] > stop
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid or stopped or expired:
                    position = 0
            elif (
                confirmed[i] != 0
                and np.isfinite(closes[i])
                and np.isfinite(atr[i])
                and atr[i] > 0.0
            ):
                position = int(confirmed[i])
                entry_bar = i + 1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    p = Params()
    index = pd.date_range("2020-01-01", periods=16)
    candles = pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close", "volume")}, index=index)
    for side in (1, -1):
        features = pd.DataFrame({"candidate": [0, 0, side, side] + [0] * 12, "atr": 1.0}, index=index)
        candles["close"] = 100.0
        candles["open"] = 100.0
        candles.iloc[4, candles.columns.get_loc("open")] = 100.0 + side
        candles.iloc[5, candles.columns.get_loc("close")] = 100.0 - side * 1.5
        features.iloc[5, features.columns.get_loc("atr")] = 20.0
        result = strategy.generate_signals(candles, features, None, p).data
        assert result["signal"].iloc[:7].tolist() == [0, 0, 0, 0, side, side, 0]
        candles["close"] = 100.0
        result = strategy.generate_signals(candles, features, None, p).data
        assert result["signal"].iloc[4:12].eq(side).all() and result["signal"].iloc[12] == 0
        features["candidate"] = 0
        features.iloc[3, 0] = side
        assert strategy.generate_signals(candles, features, None, p).data["signal"].eq(0).all()
    rng = np.random.default_rng(42)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 400)))
    history = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0}, index=pd.date_range("2021-01-01", periods=400))
    full = strategy.generate_signals(history, strategy.indicators(history, p), None, p).data
    prefix = history.iloc[:300]
    partial = strategy.generate_signals(prefix, strategy.indicators(prefix, p), None, p).data
    pd.testing.assert_frame_equal(full.iloc[:300], partial)
    assert full["signal"].isin([-1, 0, 1]).all() and full["size"].gt(0).all()
