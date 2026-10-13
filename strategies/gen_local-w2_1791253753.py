from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalEnergyParams:
    history: int = 252
    energy_window: int = 8
    atr_period: int = 14
    min_coarse_share: float = 0.55
    min_edge: float = 0.0005
    min_samples: int = 8
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[SeasonalEnergyParams]):
    strategy_id = "gen_local-w2_1791253753"

    @classmethod
    def params_type(cls):
        return SeasonalEnergyParams

    def warmup_bars(self, params: SeasonalEnergyParams) -> int:
        return int(max(params.history + params.energy_window + 11, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: SeasonalEnergyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        r = out["return"]
        for half in (1, 2, 4):
            block = r.rolling(half).sum()
            out[f"haar_{half}"] = (block - block.shift(half)) / np.sqrt(2.0 * half)
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(params.energy_window).mean()
        total = out[["energy_1", "energy_2", "energy_4"]].sum(axis=1, min_count=3)
        out["coarse_share"] = (out["energy_2"] + out["energy_4"]) / total.where(total > 0)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        # This completed two-session outcome belongs to a decision three bars ago.
        out["outcome"] = data["open"] / data["open"].shift(2).where(data["open"].shift(2) > 0) - 1.0
        out["weekday"] = data.index.dayofweek
        out["origin_weekday"] = out["weekday"].shift(3)
        out["origin_share"] = out["coarse_share"].shift(3)
        out["calendar_edge"] = np.nan
        out["calendar_abs"] = np.nan
        for day in range(5):
            sample = ((out["origin_weekday"] == day)
                      & (out["origin_share"] >= params.min_coarse_share)
                      & np.isfinite(out["outcome"]))
            out[f"sample_{day}"] = sample.astype(float)
            count = out[f"sample_{day}"].rolling(params.history).sum()
            out[f"count_{day}"] = count
            denominator = count.where(count >= params.min_samples)
            out[f"edge_{day}"] = out["outcome"].where(sample, 0.0).rolling(params.history).sum() / denominator
            out[f"abs_{day}"] = out["outcome"].abs().where(sample, 0.0).rolling(params.history).sum() / denominator
            today = out["weekday"] == day
            out.loc[today, "calendar_edge"] = out.loc[today, f"edge_{day}"]
            out.loc[today, "calendar_abs"] = out.loc[today, f"abs_{day}"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SeasonalEnergyParams,
    ) -> SignalFrame:
        edge = indicators["calendar_edge"]
        share = indicators["coarse_share"]
        strength = (edge / indicators["calendar_abs"].where(indicators["calendar_abs"] > 0)) * share
        conviction = strength.clip(0.1, 1.0).fillna(0.1).to_numpy()
        eligible = ((edge > params.min_edge) & (share >= params.min_coarse_share)
                    & (indicators["atr"] > 0) & np.isfinite(strength)).to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        signal = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        entry_bar = -1
        fixed_stop = np.nan
        held_size = 1.0
        warmup = max(self.warmup_bars(params), ctx.warmup_bars)
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Freeze the entry-open reference and the decision bar's ATR.
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
                stopped = np.isfinite(closes[i]) and closes[i] < fixed_stop
                expired = i - entry_bar + 1 >= 2
                if stopped or expired or not np.isfinite(fixed_stop):
                    entry_bar = -1
                else:
                    signal[i] = 1
                    sizes[i] = held_size
            elif i >= warmup and eligible[i] and np.isfinite(closes[i]) and closes[i] > 0:
                entry_bar = i + 1
                held_size = float(conviction[i])
                signal[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SeasonalEnergyParams()
    start = strategy.warmup_bars(params)
    index = pd.bdate_range("2020-01-01", periods=start + 6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 97.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ind = pd.DataFrame(
        {"calendar_edge": 0.0, "calendar_abs": 0.002, "coarse_share": 0.8, "atr": 1.0},
        index=index,
    )
    ind.loc[index[start], "calendar_edge"] = 0.001
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=start)
    signals = strategy.generate_signals(data, ind, ctx, params).data
    assert signals["signal"].iloc[start:start + 4].tolist() == [0, 1, 1, 0]
    assert np.isclose(signals["size"].iloc[start + 1], 0.4)
    data.loc[index[start + 1], "close"] = 98.0
    stopped = strategy.generate_signals(data, ind, ctx, params).data
    assert stopped["signal"].iloc[start:start + 3].tolist() == [0, 1, 0]
    actual = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:-2], params)
    pd.testing.assert_frame_equal(actual.iloc[:-2], prefix)
    assert (stopped["size"] > 0).all() and set(stopped["signal"]) <= {0, 1}


if __name__ == "__main__":
    _self_check()
