from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 10
    entry_threshold: float = 0.72
    atr_window: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791239053"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(16 + params.energy_window - 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if (
            params.energy_window < 1 or params.atr_window < 1
            or params.hold_bars < 1 or not 0.0 < params.entry_threshold < 1.0
            or not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0
            or not np.isfinite(params.entry_slippage_bps)
            or params.entry_slippage_bps < 0.0
        ):
            raise ValueError("Invalid wavelet strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - out["previous_close"]).abs(),
                (data["low"] - out["previous_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        for half in (2, 4, 8):
            mean = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"mean_{half}"] = mean
            # Causal, overlapping Haar coefficients: newer minus older half.
            out[f"haar_{half}"] = np.sqrt(half / 2.0) * (mean - mean.shift(half))
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = (
            out["energy_2"] + out["energy_4"] + out["energy_8"]
        )
        out["coarse_share"] = out["energy_8"] / out["total_energy"].where(
            out["total_energy"] > 0.0
        )
        out["coarse_mean"] = out["haar_8"].rolling(
            params.energy_window, min_periods=params.energy_window
        ).mean()
        out["bullish_coherence"] = (
            out["coarse_mean"] / np.sqrt(out["energy_8"].where(out["energy_8"] > 0.0))
        ).clip(0.0, 1.0)
        out["score"] = (
            out["coarse_share"] * out["bullish_coherence"]
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        count = len(data)
        decisions = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.25, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        scores = indicators["score"].fillna(0.0).to_numpy(dtype=float)
        active = False
        pending_entry = False
        stop = np.nan
        entry_bar = -1
        entry_size = 0.25
        for i in range(count):
            if pending_entry:
                # This bar's open is known now; yesterday's ATR stays frozen.
                if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                    raise ValueError("Entry open must be finite and positive")
                entry_price = opens[i] * (1.0 + params.entry_slippage_bps / 10000.0)
                stop = entry_price - params.stop_atr * atr[i - 1]
                entry_bar = i
                active = True
                pending_entry = False
            exited = False
            if active:
                if (
                    not np.isfinite(closes[i]) or closes[i] < stop
                    or i - entry_bar + 1 >= params.hold_bars
                ):
                    active = False
                    exited = True
            if (
                not active and not exited
                and i + 1 >= self.warmup_bars(params)
                and np.isfinite(closes[i]) and closes[i] > 0.0
                and np.isfinite(atr[i]) and atr[i] > 0.0
                and np.isfinite(scores[i]) and scores[i] >= params.entry_threshold
            ):
                pending_entry = True
                entry_size = float(0.25 + 0.70 * np.clip(scores[i], 0.0, 1.0))
            decisions[i] = int(active or pending_entry)
            sizes[i] = entry_size
        df = pd.DataFrame({"signal": decisions, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = WaveletParams(energy_window=1, atr_window=1, stop_atr=2.0)
    index = pd.date_range("2020-01-01", periods=26)
    close = np.full(26, 100.0)
    close[16:19] = [110.0, 99.0, 97.0]
    opening = np.full(26, 100.0)
    opening[16] = 102.0
    data = pd.DataFrame(
        {"open": opening, "high": np.maximum(opening, close) + 1.0,
         "low": np.minimum(opening, close) - 1.0, "close": close, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"score": 0.9, "atr": 2.0}, index=index)
    indicators.loc[index[17:], "atr"] = 10.0
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=16)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[:16].eq(0).all()
    assert result["signal"].iloc[16:19].eq(1).all()
    assert result["signal"].iloc[19] == 0
    assert np.isclose(result["size"].iloc[16], 0.88)
    shorter = WaveletParams(energy_window=1, atr_window=1, hold_bars=3, stop_atr=100.0)
    timed = strategy.generate_signals(data, indicators, ctx, shorter).data
    assert timed["signal"].iloc[16:19].eq(1).all()
    assert timed["signal"].iloc[19] == 0
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    derived = strategy.indicators(flat, params)
    assert derived["score"].eq(0.0).all()
    assert strategy.generate_signals(flat, derived, ctx, params).data["signal"].eq(0).all()
    rising = flat.copy()
    rising["close"] = 100.0 * np.exp(np.arange(26) * 0.003)
    rising["open"] = rising["close"]
    rising["high"] = rising["close"] + 1.0
    rising["low"] = rising["close"] - 1.0
    assert strategy.indicators(rising, params)["score"].iloc[-1] > params.entry_threshold


if __name__ == "__main__":
    _self_check()
