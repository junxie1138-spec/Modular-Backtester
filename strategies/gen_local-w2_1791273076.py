from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    expansion_ratio: float = 1.1
    trail_atr: float = 2.0

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in (self.expansion_ratio, self.trail_atr)):
            raise ValueError("Parameters must be finite and positive.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791273076"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["vol_fast"] = out["return"].rolling(5, min_periods=5).std(ddof=0)
        out["vol_slow"] = out["return"].rolling(20, min_periods=20).std(ddof=0)
        out["vol_ratio"] = out["vol_fast"] / out["vol_slow"].replace(0, np.nan)
        out["ordinal"] = np.arange(len(data), dtype=float)
        for name, energy in (
            ("up", out["return"].clip(lower=0).pow(2)),
            ("down", out["return"].clip(upper=0).pow(2)),
        ):
            out[name + "_energy"] = energy
            out[name + "_timed_energy"] = energy * out["ordinal"]
            out[name + "_mass"] = energy.rolling(20, min_periods=20).sum()
            out[name + "_moment"] = out[name + "_timed_energy"].rolling(20, min_periods=20).sum()
            out[name + "_centroid"] = (
                out[name + "_moment"] / out[name + "_mass"].replace(0, np.nan)
                - (out["ordinal"] - 19)
            ) / 19
        out["timing_lead"] = out["up_centroid"] - out["down_centroid"]
        out["displacement"] = data["close"].pct_change(10, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["setup"] = np.select([
            (out["vol_ratio"] > params.expansion_ratio)
            & (out["timing_lead"] > 0.2) & (out["displacement"] > 0),
            (out["vol_ratio"] > params.expansion_ratio)
            & (out["timing_lead"] < -0.2) & (out["displacement"] < 0),
        ], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        setup = indicators["setup"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        direction = 0
        pending = False
        entry = stop = peak = 0.0
        armed = False
        age = 0
        for i, (opening, high, low, close) in enumerate(prices):
            if i < self.warmup_bars(params):
                continue
            if not (np.isfinite(prices[i]).all() and (prices[i] > 0).all()
                    and np.isfinite(atr[i]) and atr[i] > 0):
                direction = 0
                pending = False
                continue
            if direction:
                if pending:
                    # Entry is observed on the actual next bar, never the decision bar.
                    entry = direction * opening
                    stop = entry - 2.5 * atr[i - 1]
                    peak = entry
                    age = 0
                    armed = False
                    pending = False
                age += 1
                favorable = high if direction == 1 else -low
                adverse = low if direction == 1 else -high
                settlement = direction * close
                # Signed-price coordinates make the short stop ratchet upward too.
                # Test the prior stop before using this bar's favorable excursion.
                if adverse <= stop or age >= 10:
                    direction = 0
                    continue
                peak = max(peak, favorable)
                if settlement - entry >= 0.01 * abs(entry):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    if settlement <= stop:
                        direction = 0
                        continue
                signals[i] = direction
            elif setup[i]:
                direction = int(setup[i])
                pending = True
                signals[i] = direction
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=27, freq="B")
    for direction in (1, -1):
        prices = np.tile([100.0, 100.2, 99.8, 100.0], (27, 1))
        prices[22:25] = [
            [100.0, 101.4, 99.8, 101.2],
            [101.2, 104.0, 101.0, 103.0],
            [103.0, 103.5, 101.5, 102.0],
        ]
        if direction == -1:
            prices = 200.0 - prices
            prices[:, [1, 2]] = prices[:, [2, 1]]
        data = pd.DataFrame(prices, columns=["open", "high", "low", "close"], index=index)
        data["volume"] = 1000.0
        indicators = pd.DataFrame({"atr": 1.0, "setup": 0}, index=index)
        indicators.iloc[21, indicators.columns.get_loc("setup")] = direction
        ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=21)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        expected = np.zeros(27, dtype=int)
        expected[22:25] = direction
        assert np.array_equal(result["signal"].to_numpy(), expected)
        assert (result["size"] > 0).all()
        computed = strategy.indicators(data, params)
        assert computed.index.equals(data.index)
        assert not np.isinf(computed.to_numpy(dtype=float)).any()


if __name__ == "__main__":
    _self_check()
