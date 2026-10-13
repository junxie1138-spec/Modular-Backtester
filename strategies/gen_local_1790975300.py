from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    gap_threshold: float = 0.65
    stop_atr: float = 2.25


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790975300"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return 527

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        prior_atr = out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["gap"] = (data["open"] - previous) / prior_atr
        out["phase"] = np.minimum((data.index.day - 1) // 7, 3)
        # At t, the decision at t-8 has completed eight sessions:
        # entry at open[t-7], final holding-session close at close[t].
        out["mature_gap"] = out["gap"].shift(8)
        out["mature_return"] = (
            (data["close"] - data["open"].shift(7)) / prior_atr.shift(8)
        )
        out["mature_phase"] = out["phase"].shift(8)
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out["mature_gap"].notna() & out["mature_return"].notna()
        for phase in range(4):
            selected = valid & out["mature_phase"].eq(phase)
            x = out["mature_gap"].where(selected, 0.0)
            y = out["mature_return"].where(selected, 0.0)
            count = selected.astype(float).rolling(504, min_periods=504).sum()
            denominator = count.where(count >= 40)
            mean_x = x.rolling(504, min_periods=504).sum() / denominator
            mean_y = y.rolling(504, min_periods=504).sum() / denominator
            variance = (
                x.pow(2).rolling(504, min_periods=504).sum() / denominator
                - mean_x.pow(2)
            )
            covariance = (
                (x * y).rolling(504, min_periods=504).sum() / denominator
                - mean_x * mean_y
            )
            out[f"count_{phase}"] = count
            out[f"mean_gap_{phase}"] = mean_x
            out[f"mean_return_{phase}"] = mean_y
            out[f"variance_{phase}"] = variance
            out[f"covariance_{phase}"] = covariance
            out[f"beta_{phase}"] = covariance / variance.where(variance > 1e-8)
        out["seasonal_gap"] = np.nan
        out["seasonal_beta"] = np.nan
        for phase in range(4):
            mask = out["phase"].eq(phase)
            out.loc[mask, "seasonal_gap"] = out.loc[mask, f"mean_gap_{phase}"]
            out.loc[mask, "seasonal_beta"] = out.loc[mask, f"beta_{phase}"]
        out["gap_surprise"] = out["gap"] - out["seasonal_gap"]
        out["response"] = out["seasonal_beta"] * out["gap_surprise"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        if not np.isfinite(params.gap_threshold) or params.gap_threshold <= 0:
            raise ValueError("gap_threshold must be finite and positive")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        atr = indicators["atr"].to_numpy(dtype=float)
        surprise = indicators["gap_surprise"].to_numpy(dtype=float)
        response = indicators["response"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    # The next open is read only when that session arrives.
                    # Freeze the decision-bar ATR and actual opening-price proxy.
                    stop = opens[i] - side * params.stop_atr * entry_atr
                bad_price = not np.isfinite(closes[i]) or not np.isfinite(stop)
                stopped = not bad_price and side * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= 8
                if bad_price or stopped or expired:
                    side = 0
                raw[i] = side
                continue
            ready = (
                i >= self.warmup_bars(params) - 1
                and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(surprise[i]) and np.isfinite(response[i])
            )
            if ready and abs(surprise[i]) >= params.gap_threshold and abs(response[i]) >= 0.20:
                side = 1 if response[i] > 0 else -1
                entry_bar = i + 1
                entry_atr = atr[i]
                raw[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GeneratedParams(stop_atr=2.0)
    start = strategy.warmup_bars(params)
    index = pd.bdate_range("2020-01-01", periods=start + 12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 96.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    indicator = pd.DataFrame(
        {"atr": 1.0, "gap_surprise": 0.0, "response": 0.0}, index=index
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=start)
    for side in (1, -1):
        indicator.loc[index[start - 1], ["gap_surprise", "response"]] = side
        data.loc[index[start], "close"] = 100.0 - side * 3.0
        signals = strategy.generate_signals(data, indicator, ctx, params).data["signal"]
        assert signals.iloc[start - 1] == 0
        assert signals.iloc[start] == side
        assert signals.iloc[start + 1] == 0
        data.loc[index[start], "close"] = 100.0
        signals = strategy.generate_signals(data, indicator, ctx, params).data["signal"]
        assert (signals.iloc[start:start + 8] == side).all()
        assert signals.iloc[start + 8] == 0
    complete = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:-3], params)
    assert np.allclose(
        complete.iloc[:-3].to_numpy(), prefix.to_numpy(), equal_nan=True
    )


if __name__ == "__main__":
    _self_check()
