from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    volume_window: int = 21
    echo_lag: int = 5
    correlation_window: int = 63
    percentile_window: int = 126
    correlation_quantile: float = 0.70
    volume_quantile: float = 0.75
    atr_window: int = 14
    breakeven_pct: float = 0.03
    trail_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local_1791322343"

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return int(max(
            params.volume_window + params.echo_lag
            + params.correlation_window + params.percentile_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if min(params.volume_window, params.correlation_window,
               params.percentile_window, params.atr_window) < 2:
            raise ValueError("Rolling windows must be at least two bars")
        if params.echo_lag < 1 or params.hold_bars < 1:
            raise ValueError("Lag and holding period must be positive")
        if not (0 < params.correlation_quantile < 1
                and 0 < params.volume_quantile < 1):
            raise ValueError("Quantiles must lie strictly between zero and one")
        if not (np.isfinite(params.breakeven_pct) and params.breakeven_pct > 0
                and np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("Breakeven threshold and ATR multiplier must be positive")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] > 0)
        out["previous_close"] = close.shift(1)
        out["return"] = close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        out["volume_reference"] = volume.rolling(
            params.volume_window, min_periods=params.volume_window
        ).median().shift(1)
        out["volume_threshold"] = volume.rolling(
            params.volume_window, min_periods=params.volume_window
        ).quantile(params.volume_quantile).shift(1)
        out["relative_volume"] = (volume / out["volume_reference"]).replace(
            [np.inf, -np.inf], np.nan
        )
        out["impulse"] = out["return"] * out["relative_volume"]
        out["lagged_impulse"] = out["impulse"].shift(params.echo_lag)
        out["echo_correlation"] = out["return"].rolling(
            params.correlation_window, min_periods=params.correlation_window
        ).corr(out["lagged_impulse"]).clip(-1, 1).replace(
            [np.inf, -np.inf], np.nan
        )
        out["correlation_threshold"] = out["echo_correlation"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.correlation_quantile).shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().replace([np.inf, -np.inf], np.nan)
        out["volume_confirmed"] = volume > out["volume_threshold"]
        out["lag_volume_confirmed"] = out["volume_confirmed"].shift(
            params.echo_lag, fill_value=False
        )
        out["entry_setup"] = (
            (out["echo_correlation"] > out["correlation_threshold"])
            & (out["echo_correlation"] > 0)
            & (out["return"] > 0)
            & (out["lagged_impulse"] > 0)
            & out["volume_confirmed"]
            & out["lag_volume_confirmed"]
            & (out["atr"] > 0)
        ).fillna(False)
        out["entry"] = out["entry_setup"] & ~out["entry_setup"].shift(
            1, fill_value=False
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EchoParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        pending = False
        entry_index = -1
        entry_price = stop = peak = 0.0
        armed = False

        for i in range(len(data)):
            # A decision from the previous close enters at this bar's open.
            if pending:
                pending = False
                if (i > 0 and np.isfinite(opens[i]) and opens[i] > 0
                        and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0):
                    entry_price = opens[i]
                    stop = entry_price - params.trail_atr * atrs[i - 1]
                    peak = entry_price
                    armed = False
                    entry_index = i

            if entry_index >= 0:
                # Test the existing stop before raising it from this bar's high.
                touched = np.isfinite(lows[i]) and lows[i] <= stop
                expired = i - entry_index + 1 >= params.hold_bars
                if touched or expired:
                    entry_index = -1
                    continue
                if np.isfinite(highs[i]):
                    peak = max(peak, highs[i])
                    armed = armed or peak >= entry_price * (1 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry_price)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, peak - params.trail_atr * atrs[i])
                raw[i] = 1
            elif (entries[i] and np.isfinite(closes[i]) and closes[i] > 0
                  and np.isfinite(atrs[i]) and atrs[i] > 0):
                pending = True
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # ATR expansion must never lower an already-raised stop.
    index = pd.date_range("2020-01-01", periods=7, freq="B")
    data = pd.DataFrame({
        "open": [100, 100, 104, 106, 105, 104, 104],
        "high": [100, 104, 106, 105, 104, 104, 104],
        "low": [99, 99, 103, 104, 103, 103, 103],
        "close": [100, 104, 106, 105, 104, 104, 104],
        "volume": [1000] * 7,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1, 1, 1, 3, 1, 1, 1],
        "entry": [True, False, False, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, EchoParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
