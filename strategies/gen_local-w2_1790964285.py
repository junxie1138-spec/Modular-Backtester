from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    path_window: int = 16
    vol_window: int = 20
    atr_window: int = 14
    area_threshold: float = 0.08
    target_vol: float = 0.10
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1790964285"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.path_window, params.vol_window, params.atr_window) + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.path_window, params.vol_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if not (0 < params.area_threshold < 0.5 and params.target_vol > 0
                and params.stop_atr > 0 and params.hold_bars > 0):
            raise ValueError("Invalid threshold, volatility target, stop, or holding period.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["variance_step"] = out["return"].pow(2)
        out["return_path"] = out["return"].fillna(0.0).cumsum()
        out["variance_path"] = out["variance_step"].fillna(0.0).cumsum()
        x = out["return_path"]
        y = out["variance_path"]
        r = out["return"]
        v = out["variance_step"]
        length = params.path_window
        out["cross_step"] = x.shift(1).fillna(0.0) * v - y.shift(1).fillna(0.0) * r
        out["net_return"] = r.rolling(length).sum()
        out["total_variance"] = v.rolling(length).sum()
        out["return_length"] = r.abs().rolling(length).sum()
        # Antisymmetric second-level signature of (cumulative return, variance).
        # Translate each rolling segment to its own origin before measuring area.
        out["area"] = 0.5 * (
            out["cross_step"].rolling(length).sum()
            - x.shift(length).fillna(0.0) * out["total_variance"]
            + y.shift(length).fillna(0.0) * out["net_return"]
        )
        denominator = out["return_length"] * out["total_variance"]
        out["normalized_area"] = (
            out["area"] / denominator.where(denominator > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["realized_vol"] = r.rolling(params.vol_window).std(ddof=0) * np.sqrt(252.0)
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1).where(close.notna() & previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["exposure"] = (
            params.target_vol / out["realized_vol"].where(out["realized_vol"] > 0)
        ).replace([np.inf, -np.inf], np.nan).clip(0.05, 1.0).fillna(0.05)
        out["eligible"] = (
            (out["normalized_area"] < -params.area_threshold)
            & (out["net_return"] > 0)
            & (out["realized_vol"] > 0)
            & (out["atr"] > 0)
            & close.notna()
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        raw = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        exposure = indicators["exposure"].to_numpy(dtype=float)
        active = False
        pending = False
        confirmations = 0
        held = 0
        entry_atr = 0.0
        stop = 0.0
        trade_size = 1.0
        for i in range(n):
            if pending:
                # Read the entry open only when that bar has arrived.
                # ATR and exposure were frozen at the preceding decision close.
                if np.isfinite(opens[i]) and opens[i] > 0:
                    stop = opens[i] - params.stop_atr * entry_atr
                    active = True
                    held = 0
                pending = False
            if active:
                held += 1
                stopped = not np.isfinite(closes[i]) or closes[i] < stop
                if stopped or held >= params.hold_bars:
                    active = False
                    confirmations = 0
                else:
                    raw[i] = 1
                    sizes[i] = trade_size
                continue
            valid = (eligible[i] and np.isfinite(atrs[i]) and atrs[i] > 0
                     and np.isfinite(exposure[i]) and exposure[i] > 0)
            confirmations = confirmations + 1 if valid else 0
            if confirmations >= 2:
                pending = True
                entry_atr = atrs[i]
                trade_size = float(np.clip(exposure[i], 0.05, 1.0))
                raw[i] = 1
                sizes[i] = trade_size
                confirmations = 0
        df = pd.DataFrame({"signal": raw, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams(path_window=3, vol_window=3, atr_window=3, stop_atr=2.0)
    index = pd.date_range("2020-01-01", periods=25)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"atr": 1.0, "eligible": True, "exposure": 0.5}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=5)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[:2].eq(0).all()
    assert result["signal"].iloc[2:20].eq(1).all()
    assert result["signal"].iloc[20] == 0
    assert result["size"].iloc[2] == 0.5
    data.loc[index[2], ["open", "close"]] = [110.0, 107.0]
    indicators.loc[index[2], "atr"] = 100.0
    stopped = strategy.generate_signals(data, indicators, ctx, params).data
    assert stopped["signal"].iloc[2] == 1 and stopped["signal"].iloc[3] == 0
    returns = np.array([0.01, -0.04, 0.02, 0.015, 0.01])
    sample = data.iloc[:6].copy()
    sample["close"] = 100.0 * np.exp(np.r_[0.0, returns.cumsum()])
    derived = strategy.indicators(sample, params)
    r = returns[-3:]
    expected = 0.5 * sum(
        r[a] * r[b] ** 2 - r[a] ** 2 * r[b]
        for a in range(3) for b in range(a + 1, 3)
    )
    assert np.isclose(derived["area"].iloc[-1], expected, atol=1e-12)
    flat = strategy.indicators(data.assign(open=100.0, close=100.0), params)
    assert not flat["eligible"].any() and flat["exposure"].gt(0).all()


if __name__ == "__main__":
    _self_check()
