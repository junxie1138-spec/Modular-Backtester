from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletAnalogParams:
    archive_bars: int = 504
    analog_percentile: float = 0.15
    entry_percentile: float = 0.70
    hold_bars: int = 18
    stop_atr: float = 2.5

    def __post_init__(self):
        if self.archive_bars < 128 or self.hold_bars < 1:
            raise ValueError("archive_bars must be >=128 and hold_bars positive")
        if not 0 < self.analog_percentile <= 1:
            raise ValueError("analog_percentile must be in (0, 1]")
        if not 0 < self.entry_percentile < 1 or not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("Invalid entry percentile or stop multiplier")


class GeneratedStrategy(BaseStrategy[WaveletAnalogParams]):
    strategy_id = "gen_local_1790876681"

    @classmethod
    def params_type(cls):
        return WaveletAnalogParams

    def warmup_bars(self, params: WaveletAnalogParams) -> int:
        return int(params.archive_bars + params.hold_bars + 32 + 127)

    def indicators(self, data: pd.DataFrame, params: WaveletAnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        energy_columns = []
        feature_columns = []
        for half in (1, 2, 4, 8):
            mean = out["log_return"].rolling(half, min_periods=half).mean()
            coefficient = f"haar_{half}"
            energy = f"energy_{half}"
            share = f"share_{half}"
            out[coefficient] = (mean - mean.shift(half)) * np.sqrt(half / 2.0)
            out[energy] = out[coefficient].pow(2).rolling(16, min_periods=16).mean()
            energy_columns.append(energy)
            feature_columns.append(share)
        out["total_energy"] = out[energy_columns].sum(axis=1, min_count=4)
        denominator = out["total_energy"].where(out["total_energy"] > 0)
        for half, energy in zip((1, 2, 4, 8), energy_columns):
            out[f"share_{half}"] = out[energy] / denominator
        out["drift"] = (
            out["log_return"].rolling(16, min_periods=16).sum()
            / (out["log_return"].rolling(16, min_periods=16).std(ddof=0)
               .replace(0, np.nan) * 4.0)
        ).clip(-3, 3) / 3.0
        features = out[feature_columns + ["drift"]].to_numpy(dtype=float)
        prices = close.to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        forecasts = np.full(len(data), np.nan)
        # ponytail: bounded O(bars * archive_bars) scan; index neighbors if archives grow.
        for i in range(32 + params.hold_bars + 64, len(data)):
            if not np.isfinite(features[i]).all():
                continue
            # Labels end strictly before today's bar; never consult future outcomes.
            end = i - params.hold_bars
            candidates = np.arange(max(32, end - params.archive_bars), end)
            valid = (
                np.isfinite(features[candidates]).all(axis=1)
                & np.isfinite(prices[candidates + params.hold_bars])
                & np.isfinite(opens[candidates + 1])
                & (opens[candidates + 1] > 0)
            )
            candidates = candidates[valid]
            if len(candidates) < 64:
                continue
            distances = np.square(features[candidates] - features[i]).sum(axis=1)
            cutoff = np.quantile(distances, params.analog_percentile)
            neighbors = candidates[distances <= cutoff]
            outcomes = prices[neighbors + params.hold_bars] / opens[neighbors + 1] - 1.0
            forecasts[i] = outcomes.mean()
        out["forecast"] = forecasts
        out["entry_threshold"] = (
            out["forecast"].shift(1).rolling(126, min_periods=126)
            .quantile(params.entry_percentile)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveletAnalogParams
    ) -> SignalFrame:
        entry = (
            (indicators["forecast"] > indicators["entry_threshold"])
            & (indicators["forecast"] > 0)
            & (indicators["atr"] > 0)
            & np.isfinite(indicators["atr"])
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        stop = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if pending:
                # The previous close requested entry; this bar's open is its reference fill.
                if np.isfinite(opens[i]) and opens[i] > 0:
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                    entry_bar = i
                    active = True
                pending = False
            if active:
                if not np.isfinite(closes[i]) or closes[i] < stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif entry[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = True
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveletAnalogParams()
    index = pd.date_range("2020-01-01", periods=48, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index
    )
    indicators = pd.DataFrame(
        {"forecast": np.nan, "entry_threshold": 0.0, "atr": 2.0}, index=index
    )
    indicators.iloc[0, 0] = 1.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19:].eq(0).all()
    data.loc[index[2], "close"] = 120.0
    data.loc[index[3], "close"] = 99.0
    data.loc[index[4], "close"] = 94.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[1:5].eq(1).all()
    assert result["signal"].iloc[5:].eq(0).all()
    assert result["size"].gt(0).all()
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 900)))
    history = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1000.0},
        index=pd.date_range("2018-01-01", periods=900, freq="B")
    )
    full = strategy.indicators(history, params)
    prefix = strategy.indicators(history.iloc[:800], params)
    pd.testing.assert_frame_equal(full.iloc[:800], prefix)
    signals = strategy.generate_signals(history, full, None, params).data
    shorter = strategy.generate_signals(history.iloc[:800], prefix, None, params).data
    pd.testing.assert_frame_equal(signals.iloc[:800], shorter)
    assert signals["signal"].isin([0, 1]).all()


if __name__ == "__main__":
    _self_check()
