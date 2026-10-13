from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalPulseParams:
    fd_order: float = 0.6
    fd_lags: int = 24
    baseline_window: int = 20
    rank_window: int = 126
    entry_quantile: float = 0.8
    vol_window: int = 20
    target_vol: float = 0.12
    atr_window: int = 14
    arm_gain: float = 0.025
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[FractionalPulseParams]):
    strategy_id = "gen_local_1791300837"

    @classmethod
    def params_type(cls):
        return FractionalPulseParams

    def warmup_bars(self, params: FractionalPulseParams) -> int:
        return int(max(params.fd_lags + params.baseline_window + params.rank_window + 1,
                       params.vol_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: FractionalPulseParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        weights = np.empty(params.fd_lags, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.fd_lags):
            weights[k] = weights[k - 1] * (k - 1 - params.fd_order) / k
        fd = pd.Series(0.0, index=data.index)
        for k, weight in enumerate(weights):
            fd = fd + weight * close.shift(k)
        prev = close.shift(1)
        tr = pd.concat([data["high"] - data["low"],
                        (data["high"] - prev).abs(),
                        (data["low"] - prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        baseline = fd.rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        score = ((fd - baseline) / atr.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        threshold = score.rolling(params.rank_window, min_periods=params.rank_window).quantile(params.entry_quantile).shift(1)
        vol = close.pct_change().rolling(params.vol_window, min_periods=params.vol_window).std() * np.sqrt(252.0)
        return pd.DataFrame({"fd": fd, "true_range": tr, "atr": atr,
                             "baseline": baseline, "score": score,
                             "threshold": threshold, "vol": vol}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: FractionalPulseParams) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        scores = indicators["score"].to_numpy(dtype=float)
        thresholds = indicators["threshold"].to_numpy(dtype=float)
        vols = indicators["vol"].to_numpy(dtype=float)
        long = False
        entry = np.nan
        stop = -np.inf
        held = 0
        for i in range(len(data)):
            if not long:
                if (np.isfinite(scores[i]) and np.isfinite(thresholds[i])
                        and np.isfinite(vols[i]) and vols[i] > 0
                        and scores[i] > thresholds[i]):
                    long = True
                    entry = np.nan
                    stop = -np.inf
                    held = 0
                    signal[i] = 1
            else:
                if np.isnan(entry):
                    entry = opens[i]
                held += 1
                if highs[i] >= entry * (1.0 + params.arm_gain) and np.isfinite(atrs[i]):
                    stop = max(stop, entry, highs[i] - params.trail_atr * atrs[i])
                if closes[i] <= stop or held >= params.max_hold:
                    long = False
                else:
                    signal[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = (params.target_vol / indicators["vol"].replace(0.0, np.nan)).clip(lower=0.05, upper=1.0).fillna(1.0)
        return SignalFrame(data=df, signal_column="signal", size_column="size")

if __name__ == "__main__":
    sample = pd.DataFrame({
        "open": [100.0] * 5, "high": [100.0, 100.0, 104.0, 102.0, 100.0],
        "close": [100.0, 100.0, 103.0, 101.0, 100.0],
    }, index=pd.date_range("2020-01-01", periods=5))
    features = pd.DataFrame({
        "score": [1.0, 0.0, 0.0, 0.0, 0.0],
        "threshold": [0.5] * 5, "vol": [0.1] * 5, "atr": [1.0] * 5,
    }, index=sample.index)
    context = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = GeneratedStrategy().generate_signals(sample, features, context, FractionalPulseParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0]
