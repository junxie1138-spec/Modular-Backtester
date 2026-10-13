from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    vol_window: int = 20
    volume_window: int = 10
    atr_window: int = 14
    shock_multiple: float = 2.0
    volume_multiple: float = 1.1
    target_vol: float = 0.12
    stop_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791252112"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.vol_window + 10, params.volume_window + 1, params.atr_window + 1, 3))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.vol_window, params.volume_window, params.atr_window) < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five")
        if not all(np.isfinite(x) and x > 0 for x in (params.shock_multiple, params.volume_multiple, params.target_vol, params.stop_atr)):
            raise ValueError("Thresholds and volatility targets must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["daily_vol"] = out["return"].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0)
        out["prior_vol"] = out["daily_vol"].shift(1)
        out["shock"] = ((out["prior_vol"] > 0) & (out["return"] < -params.shock_multiple * out["prior_vol"])).astype(int)
        positions = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        out["latest_shock"] = positions.where(out["shock"].eq(1)).rolling(9, min_periods=1).max()
        out["shock_age"] = positions - out["latest_shock"]
        out["two_day_return"] = out["return"].rolling(2, min_periods=2).sum()
        out["return_gate"] = out["shock_age"].between(3, 8) & out["two_day_return"].gt(0) & out["return"].gt(0)
        out["prior_volume"] = data["volume"].rolling(params.volume_window, min_periods=params.volume_window).mean().shift(1)
        out["volume_ratio"] = data["volume"].div(out["prior_volume"].where(out["prior_volume"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["volume_gate"] = out["volume_ratio"].ge(params.volume_multiple)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([(data["high"] - data["low"]).abs(), (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["annual_vol"] = out["daily_vol"] * np.sqrt(252.0)
        out["target_size"] = (params.target_vol / out["annual_vol"].where(out["annual_vol"] > 0)).clip(0.05, 0.95)
        out["entry"] = (out["return_gate"] & out["volume_gate"] & out["atr"].gt(0) & out["target_size"].notna()).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.05, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atrs = indicators["atr"].to_numpy(dtype=float)
        targets = indicators["target_size"].to_numpy(dtype=float)
        active = False
        decision_bar = -1
        stop = np.nan
        entry_atr = np.nan
        position_size = 0.05
        for i in range(n):
            if active:
                if i == decision_bar + 1:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        continue
                    # Anchor once to the execution bar's open and the decision bar's ATR.
                    stop = opens[i] - params.stop_atr * entry_atr
                if not np.isfinite(closes[i]) or closes[i] < stop or i - decision_bar >= params.hold_bars:
                    active = False
                    continue
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0 and np.isfinite(targets[i]) and targets[i] > 0:
                active = True
                decision_bar = i
                entry_atr = atrs[i]
                position_size = float(np.clip(targets[i], 0.05, 0.95))
            if active:
                signals[i] = 1
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
    data.iloc[2, data.columns.get_loc("close")] = 97.0
    indicators = pd.DataFrame({"entry": False, "atr": 1.0, "target_size": 0.5}, index=index)
    indicators.iloc[[0, 6], indicators.columns.get_loc("entry")] = True
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, RecoveryParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    assert result.loc[result["signal"].eq(1), "size"].eq(0.5).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
