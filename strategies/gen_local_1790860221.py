from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 20
    dominance: float = 0.45
    correlation_floor: float = 0.10
    profit_target: float = 0.018
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790860221"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return max(200, params.energy_window + 8)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.energy_window < 3 or not 0 < params.dominance < 1:
            raise ValueError("Invalid wavelet energy parameters")
        if not 0 < params.correlation_floor < 1:
            raise ValueError("correlation_floor must be between zero and one")
        if params.profit_target <= 0 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Positive target and three to five holding bars required")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["return"] = np.log(close).diff()
        # Causal, undecimated Haar details compare adjacent return blocks.
        for scale in (1, 2, 4):
            out[f"sum_{scale}"] = out["return"].rolling(scale, min_periods=scale).sum()
            out[f"haar_{scale}"] = (
                out[f"sum_{scale}"] - out[f"sum_{scale}"].shift(scale)
            ) / np.sqrt(2.0 * scale)
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out[["energy_1", "energy_2", "energy_4"]].sum(
            axis=1, min_count=3
        ).replace(0.0, np.nan)
        out["fine_share"] = out["energy_1"] / out["total_energy"]
        out["coarse_share"] = out["energy_4"] / out["total_energy"]
        out["lagged_return"] = out["return"].shift(1)
        out["rho"] = out["return"].rolling(
            params.energy_window, min_periods=params.energy_window
        ).corr(out["lagged_return"]).clip(-1.0, 1.0)
        persistent = (
            (out["coarse_share"] >= params.dominance)
            & (out["rho"] >= params.correlation_floor)
        )
        alternating = (
            (out["fine_share"] >= params.dominance)
            & (out["rho"] <= -params.correlation_floor)
        )
        out["forecast"] = np.select(
            [persistent, alternating],
            [out["sum_4"], -out["return"]],
            default=0.0,
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[["ma200", "fine_share", "coarse_share", "rho", "forecast"]].notna().all(axis=1)
        out["entry"] = np.select(
            [
                ready & (close > out["ma200"]) & (out["forecast"] > 0),
                ready & (close < out["ma200"]) & (out["forecast"] < 0),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                # Read the entry open only when its bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                gain = side * (closes[i] / entry_price - 1.0) if (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i])
                ) else np.nan
                expired = i - entry_bar + 1 >= params.hold_bars
                # Close-confirmed target; the exit executes next bar.
                if gain >= params.profit_target or expired:
                    side = 0
                    entry_price = np.nan
                    continue
            elif entries[i] and i + 1 < len(data):
                side = int(entries[i])
                entry_bar = i + 1
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    for side in (1, -1):
        data = pd.DataFrame(
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0},
            index=index,
        )
        ind = pd.DataFrame({"entry": [side, 0, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, side, side, side, side, 0, 0]
        data.loc[index[1], "close"] = 100.0 * (1.0 + side * 0.03)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, side, 0, 0, 0, 0, 0]
        assert (result["size"] > 0).all()
    flat = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0},
        index=pd.date_range("2020-01-01", periods=220, freq="D"),
    )
    assert strategy.indicators(flat, params)["entry"].eq(0).all()


if __name__ == "__main__":
    _self_check()
