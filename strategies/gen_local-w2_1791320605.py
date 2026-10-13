from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletPositionParams:
    channel_bars: int = 40
    energy_bars: int = 12
    entropy_floor: float = 0.90
    edge_fraction: float = 0.18
    profit_target: float = 0.035
    max_hold_bars: int = 18

    def __post_init__(self):
        if self.channel_bars < 2 or self.energy_bars < 2 or self.max_hold_bars < 1:
            raise ValueError("Lookbacks must be >= 2 and holding period >= 1.")
        if not 0.0 < self.entropy_floor <= 1.0:
            raise ValueError("entropy_floor must lie in (0, 1].")
        if not 0.0 < self.edge_fraction < 0.5:
            raise ValueError("edge_fraction must lie in (0, 0.5).")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive.")


class GeneratedStrategy(BaseStrategy[WaveletPositionParams]):
    strategy_id = "gen_local-w2_1791320605"

    @classmethod
    def params_type(cls) -> type[WaveletPositionParams]:
        return WaveletPositionParams

    def warmup_bars(self, params: WaveletPositionParams) -> int:
        # Sixteen observations for the coarsest Haar coefficient, then energy smoothing.
        return int(max(params.channel_bars, 16 + params.energy_bars - 1) + 1)

    def indicators(self, data: pd.DataFrame, params: WaveletPositionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        energy_columns = []
        for half in (2, 4, 8):
            mean_name = f"mean_{half}"
            detail_name = f"haar_{2 * half}"
            energy_name = f"energy_{2 * half}"
            out[mean_name] = out["log_close"].rolling(half, min_periods=half).mean()
            out[detail_name] = np.sqrt(half / 2.0) * (out[mean_name] - out[mean_name].shift(half))
            out[energy_name] = out[detail_name].pow(2).rolling(
                params.energy_bars, min_periods=params.energy_bars
            ).mean()
            energy_columns.append(energy_name)
        out["total_energy"] = out[energy_columns].sum(axis=1, min_count=3)
        shares = out[energy_columns].div(out["total_energy"].where(out["total_energy"] > 0.0), axis=0)
        for column in energy_columns:
            out[f"share_{column}"] = shares[column]
        # Zero energy shares contribute zero; missing energies remain missing.
        terms = shares * np.log(shares.clip(lower=1e-15))
        out["energy_entropy"] = -terms.sum(axis=1, min_count=3) / np.log(3.0)
        out["channel_low"] = data["low"].rolling(params.channel_bars, min_periods=params.channel_bars).min()
        out["channel_high"] = data["high"].rolling(params.channel_bars, min_periods=params.channel_bars).max()
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["relative_position"] = (data["close"] - out["channel_low"]) / out["channel_width"].where(out["channel_width"] > 0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletPositionParams,
    ) -> SignalFrame:
        ready = indicators["energy_entropy"].ge(params.entropy_floor)
        position_in_channel = indicators["relative_position"]
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[ready & position_in_channel.le(params.edge_fraction)] = 1
        candidate.loc[ready & position_in_channel.ge(1.0 - params.edge_fraction)] = -1
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                if entry_bar == -1:
                    # The previous close's entry decision fills at this bar's open.
                    entry_price = opens[i]
                    entry_bar = i
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        position = 0
                        entry_bar = -1
                        continue
                gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                    position = 0
                    entry_bar = -1
                    # Exit occupies this decision bar; entry requires a later confirmation.
                    continue
            elif confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(confirmed[i])
                entry_bar = -1
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = WaveletPositionParams(max_hold_bars=3)
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 105.0, "low": 95.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    for direction, edge in ((1, 0.1), (-1, 0.9)):
        ind = pd.DataFrame({
            "energy_entropy": 1.0,
            "relative_position": [edge, edge, 0.5, 0.5, 0.5, 0.5],
        }, index=index)
        result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, direction, 0]
        profitable = data.copy()
        profitable.iloc[3, profitable.columns.get_loc("close")] = 100.0 * (1.0 + direction * 0.04)
        result = GeneratedStrategy.generate_signals(None, profitable, ind, None, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, 0, 0]
        ind.iloc[1, ind.columns.get_loc("relative_position")] = 0.5
        result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
        assert not result["signal"].any()
    flat = GeneratedStrategy.indicators(None, data, params)
    assert flat["energy_entropy"].isna().all()


if __name__ == "__main__":
    _self_check()
