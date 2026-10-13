from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fast_window: int = 5
    vol_window: int = 20
    response_window: int = 63
    atr_window: int = 14
    shock_threshold: float = 0.8
    correlation_floor: float = 0.12
    target_vol: float = 0.12
    stop_atr: float = 2.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791219172"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(max(params.fast_window, params.vol_window) + params.response_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.fast_window, params.vol_window, params.response_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if params.fast_window >= params.vol_window or params.hold_bars < 1:
            raise ValueError("Require fast_window < vol_window and hold_bars >= 1.")
        if not all(np.isfinite(x) and x > 0 for x in (params.shock_threshold, params.correlation_floor, params.target_vol, params.stop_atr)):
            raise ValueError("Thresholds, target volatility and stop multiplier must be positive and finite.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["fast_vol"] = out["return"].rolling(params.fast_window, min_periods=params.fast_window).std()
        out["slow_vol"] = out["return"].rolling(params.vol_window, min_periods=params.vol_window).std()
        out["shock"] = np.log(out["fast_vol"].replace(0, np.nan) / out["slow_vol"].replace(0, np.nan))
        out["lag_shock"] = out["shock"].shift(1)
        window = out["lag_shock"].rolling(params.response_window, min_periods=params.response_window)
        out["shock_mean"] = window.mean()
        out["shock_std"] = window.std()
        out["response_std"] = out["return"].rolling(params.response_window, min_periods=params.response_window).std()
        out["response_cov"] = window.cov(out["return"])
        out["correlation"] = (out["response_cov"] / (out["shock_std"] * out["response_std"]).replace(0, np.nan)).clip(-1.0, 1.0)
        out["shock_z"] = (out["shock"] - out["shock_mean"]) / out["shock_std"].replace(0, np.nan)
        out["forecast"] = out["correlation"] * out["shock_z"]
        out["annual_vol"] = out["slow_vol"] * np.sqrt(252.0)
        out["risk_fraction"] = (params.target_vol / out["annual_vol"].replace(0, np.nan)).clip(0.05, 0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame({"signal": np.zeros(len(data), dtype=int), "size": np.full(len(data), 0.05)}, index=data.index)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        forecasts = indicators["forecast"].to_numpy(dtype=float)
        shocks = indicators["shock_z"].to_numpy(dtype=float)
        correlations = indicators["correlation"].to_numpy(dtype=float)
        risk = indicators["risk_fraction"].to_numpy(dtype=float)
        ready = np.isfinite(np.column_stack([atr, forecasts, shocks, correlations, risk])).all(axis=1) & (atr > 0)
        side, age = 0, 0
        pending = False
        entry_price = stop_distance = conviction = 0.0
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if side:
                if pending:
                    entry_price = opens[i]
                    pending = False
                age += 1
                stopped = side * (closes[i] - entry_price) < -stop_distance
                if not ready[i] or not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(closes[i]) or stopped or age >= params.hold_bars:
                    side = 0
            elif i >= warmup - 1 and ready[i] and np.isfinite(closes[i]) and closes[i] > 0:
                if abs(shocks[i]) >= params.shock_threshold and abs(correlations[i]) >= params.correlation_floor and forecasts[i] != 0:
                    side = 1 if forecasts[i] > 0 else -1
                    stop_distance = params.stop_atr * atr[i]
                    conviction = float(np.clip(abs(forecasts[i]), 0.1, 1.0))
                    age = 0
                    pending = True
            df.iat[i, 0] = side
            if side:
                df.iat[i, 1] = float(np.clip(risk[i] * conviction, 0.05, 0.95))
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams()
    count = strategy.warmup_bars(params) + 12
    index = pd.date_range("2020-01-01", periods=count, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not np.isinf(derived.to_numpy()).any()
    entry = strategy.warmup_bars(params) - 1
    for side in (1, -1):
        sample = data.copy()
        features = pd.DataFrame({"atr": 2.0, "forecast": side * 0.5, "shock_z": 1.0, "correlation": side * 0.5, "risk_fraction": 0.8}, index=index)
        features.loc[index[entry + 1]:, "atr"] = 100.0
        sample.loc[index[entry + 2], "close"] = 94.0 if side == 1 else 106.0
        result = strategy.generate_signals(sample, features, None, params).data
        assert result["signal"].iloc[entry] == 0
        assert result["signal"].iloc[entry + 1] == side
        assert result["signal"].iloc[entry + 2] == side
        assert result["signal"].iloc[entry + 3] == 0
        assert np.isclose(result["size"].iloc[entry + 1], 0.4)
        result = strategy.generate_signals(data, features, None, params).data
        assert result["signal"].iloc[entry + params.hold_bars] == side
        assert result["signal"].iloc[entry + params.hold_bars + 1] == 0
        assert set(result["signal"].unique()) <= {-1, 0, 1}
        assert result["size"].between(0.05, 0.95).all()


if __name__ == "__main__":
    _self_check()
