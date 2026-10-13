from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalInnovationParams:
    seasonal_window: int = 756
    level_window: int = 8
    slope_window: int = 20
    volatility_window: int = 63
    seasonal_edge: float = 0.002
    innovation_threshold: float = 0.10
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[SeasonalInnovationParams]):
    strategy_id = "gen_local-w2_1790992322"

    @classmethod
    def params_type(cls) -> type[SeasonalInnovationParams]:
        return SeasonalInnovationParams

    @staticmethod
    def warmup_bars(params: SeasonalInnovationParams) -> int:
        return int(max(
            params.seasonal_window + 1,
            params.level_window + params.slope_window + 2,
            params.volatility_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalInnovationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["phase"] = 2.0 * np.pi * (data.index.dayofyear - 1) / 365.25
        out["cos_phase"] = np.cos(out["phase"])
        out["sin_phase"] = np.sin(out["phase"])
        out["cos_return"] = out["return"] * out["cos_phase"]
        out["sin_return"] = out["return"] * out["sin_phase"]
        window = params.seasonal_window
        out["mean_return"] = out["return"].rolling(window, min_periods=window).mean()
        out["mean_cos"] = out["cos_phase"].rolling(window, min_periods=window).mean()
        out["mean_sin"] = out["sin_phase"].rolling(window, min_periods=window).mean()
        out["seasonal_cos"] = 2.0 * (
            out["cos_return"].rolling(window, min_periods=window).mean()
            - out["mean_return"] * out["mean_cos"]
        )
        out["seasonal_sin"] = 2.0 * (
            out["sin_return"].rolling(window, min_periods=window).mean()
            - out["mean_return"] * out["mean_sin"]
        )
        omega = 2.0 * np.pi / 252.0
        advance = omega * params.max_hold
        out["seasonal_gain"] = (
            out["seasonal_cos"] * (np.sin(out["phase"] + advance) - out["sin_phase"])
            + out["seasonal_sin"] * (out["cos_phase"] - np.cos(out["phase"] + advance))
        ) / omega
        out["latent_level"] = out["log_close"].rolling(
            params.level_window, min_periods=params.level_window
        ).mean()
        out["latent_slope"] = out["latent_level"].diff()
        out["expected_slope"] = out["latent_slope"].shift(1).rolling(
            params.slope_window, min_periods=params.slope_window
        ).mean()
        out["volatility"] = out["return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out["slope_innovation"] = (
            (out["latent_slope"] - out["expected_slope"])
            / (out["volatility"] / np.sqrt(params.level_window)).replace(0.0, np.nan)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalInnovationParams,
    ) -> SignalFrame:
        entry = (
            (indicators["seasonal_gain"] > params.seasonal_edge)
            & (indicators["slope_innovation"] > params.innovation_threshold)
            & (indicators["latent_slope"] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        entry[:GeneratedStrategy.warmup_bars(params)] = False
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            # A prior close's entry decision fills at this bar's open.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    active = True
                    entry_price = opens[i]
                    entry_bar = i
            if active:
                profit_exit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_exit = i - entry_bar + 1 >= params.max_hold
                if profit_exit or time_exit:
                    active = False
                    continue
                decisions[i] = 1
            elif entry[i] and i + 1 < len(data):
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
