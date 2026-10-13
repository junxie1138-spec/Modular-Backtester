from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationParams:
    level_span: int = 10
    slope_span: int = 6
    innovation_window: int = 24
    regime_window: int = 20
    atr_window: int = 14
    efficiency_threshold: float = 0.35
    entry_z: float = 1.5
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[InnovationParams]):
    strategy_id = "gen_local-w2_1791329014"

    @classmethod
    def params_type(cls):
        return InnovationParams

    def warmup_bars(self, params: InnovationParams) -> int:
        return int(max(
            params.level_span + params.slope_span + params.innovation_window + 3,
            params.regime_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: InnovationParams) -> pd.DataFrame:
        p = params
        if min(p.level_span, p.slope_span, p.innovation_window,
               p.regime_window, p.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if not (0 < p.efficiency_threshold < 1 and p.entry_z > 0
                and p.breakeven_pct > 0 and p.trail_atr > 0 and p.max_hold >= 1):
            raise ValueError("Invalid thresholds or holding period.")
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)
        out["change"] = close.diff()
        out["level"] = close.ewm(
            span=p.level_span, adjust=False, min_periods=p.level_span
        ).mean()
        out["observed_slope"] = out["level"].diff()
        out["latent_slope"] = out["observed_slope"].ewm(
            span=p.slope_span, adjust=False, min_periods=p.slope_span
        ).mean()
        out["innovation"] = out["observed_slope"] - out["latent_slope"].shift(1)
        out["innovation_scale"] = out["innovation"].rolling(
            p.innovation_window, min_periods=p.innovation_window
        ).std(ddof=0).shift(1)
        out["z"] = out["innovation"].div(
            out["innovation_scale"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out["travel"] = out["change"].abs().rolling(
            p.regime_window, min_periods=p.regime_window
        ).sum()
        out["efficiency"] = close.diff(p.regime_window).abs().div(
            out["travel"].replace(0.0, np.nan)
        )
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            p.atr_window, min_periods=p.atr_window
        ).mean()
        trending = out["efficiency"] >= p.efficiency_threshold
        out["score"] = out["z"].where(trending, -out["z"])
        aligned = (trending & (out["z"] * out["latent_slope"] > 0)) | (
            ~trending & (out["z"] * (close - out["level"]) > 0)
        )
        ready = np.isfinite(out[[
            "z", "latent_slope", "efficiency", "atr"
        ]]).all(axis=1) & (out["atr"] > 0) & (close > 0)
        out["candidate"] = np.sign(out["score"]).where(
            ready & aligned & (out["score"].abs() >= p.entry_z), 0
        ).fillna(0).astype(int)
        out["conviction"] = (
            out["score"].abs() / (2.0 * p.entry_z)
        ).clip(0.25, 1.0).fillna(0.25)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: InnovationParams,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidate = indicators["candidate"].to_numpy(dtype=int)
        conviction = indicators["conviction"].to_numpy(dtype=float)
        side = 0
        pending = False
        armed = False
        entry = stop = entry_atr = 0.0
        held_size = 1.0
        entry_bar = 0
        for i in range(n):
            if side:
                if pending:
                    # The previous close's instruction fills at today's open.
                    if not np.isfinite(opening[i]) or opening[i] <= 0:
                        side = 0
                        pending = False
                        continue
                    entry = opening[i]
                    entry_bar = i
                    # Signed-price stops only rise, for both long and short.
                    stop = side * entry - params.trail_atr * entry_atr
                    pending = False
                adverse = low[i] if side == 1 else -high[i]
                expired = i - entry_bar + 1 >= params.max_hold
                if (np.isfinite(adverse) and adverse <= stop) or expired:
                    side = 0
                    armed = False
                    # Exit instruction fills next bar; prohibit same-close reentry.
                    continue
                if np.isfinite(closing[i]):
                    gain = side * (closing[i] - entry) / entry
                    armed = armed or gain >= params.breakeven_pct
                    if armed:
                        stop = max(stop, side * entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            # Ratchet at the close, effective from the next bar.
                            stop = max(
                                stop, side * closing[i] - params.trail_atr * atr[i]
                            )
            elif i >= self.warmup_bars(params) and candidate[i]:
                if not np.isfinite(atr[i]) or atr[i] <= 0:
                    continue
                side = int(candidate[i])
                entry_atr = atr[i]
                held_size = float(np.clip(conviction[i], 0.25, 1.0))
                armed = False
                pending = True
            signal[i] = side
            sizes[i] = held_size if side else 1.0
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = InnovationParams()
    j = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=j + 6, freq="D")
    for direction in (1, -1):
        data = pd.DataFrame({
            "open": 100.0, "high": 100.5, "low": 99.5,
            "close": 100.0, "volume": 1000.0,
        }, index=index)
        for offset, values in enumerate([
            (102.5, 99.5, 102.0),
            (104.5, 101.0, 104.0),
            (103.0, 101.5, 102.0),
        ], start=1):
            h, l, c = values
            if direction == -1:
                h, l, c = 200.0 - l, 200.0 - h, 200.0 - c
            data.loc[index[j + offset], ["high", "low", "close"]] = [h, l, c]
        indicators = pd.DataFrame({
            "atr": 1.0, "candidate": 0, "conviction": 0.6,
        }, index=index)
        indicators.loc[index[j], "candidate"] = direction
        ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=j)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].iloc[j:j + 5].tolist() == [
            0, direction, direction, direction, 0
        ]
        assert np.allclose(result["size"].iloc[j + 1:j + 4], 0.6)
        assert (result["size"] > 0).all()
        actual = strategy.indicators(data, params)
        assert actual.index.equals(data.index)
        assert actual["candidate"].isin([-1, 0, 1]).all()
        assert (actual["conviction"] > 0).all()


if __name__ == "__main__":
    _self_check()
