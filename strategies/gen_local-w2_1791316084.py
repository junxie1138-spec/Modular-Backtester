from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalVolumeParams:
    season_window: int = 252
    volume_window: int = 20
    atr_window: int = 14
    outcome_bars: int = 5
    min_samples: int = 6
    volume_threshold: float = 1.15
    move_atr: float = 0.35
    min_edge: float = 0.0015
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    initial_stop_atr: float = 2.0
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[SeasonalVolumeParams]):
    strategy_id = "gen_local-w2_1791316084"

    @classmethod
    def params_type(cls):
        return SeasonalVolumeParams

    def warmup_bars(self, params: SeasonalVolumeParams) -> int:
        return int(max(params.volume_window + 1, params.atr_window + 1) + params.outcome_bars + params.season_window + 1)

    def indicators(self, data: pd.DataFrame, params: SeasonalVolumeParams) -> pd.DataFrame:
        p = params
        if min(p.season_window, p.volume_window, p.atr_window, p.outcome_bars, p.min_samples, p.max_hold) < 1:
            raise ValueError("Lookbacks, sample counts and holding periods must be positive")
        if min(p.volume_threshold, p.move_atr, p.min_edge, p.breakeven_pct, p.trail_atr, p.initial_stop_atr) <= 0:
            raise ValueError("Signal and stop thresholds must be positive")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([(data["high"] - data["low"]), (data["high"] - previous).abs(), (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_window, min_periods=p.atr_window).mean().replace(0, np.nan)
        out["volume_baseline"] = data["volume"].rolling(p.volume_window, min_periods=p.volume_window).mean().shift(1).replace(0, np.nan)
        out["relative_volume"] = data["volume"] / out["volume_baseline"]
        out["move"] = data["close"].diff()
        out["move_strength"] = out["move"].abs() / out["atr"]
        out["weekday"] = data.index.dayofweek
        valid = (out["relative_volume"] >= p.volume_threshold) & (out["move_strength"] >= p.move_atr)
        out["event_direction"] = np.sign(out["move"]).where(valid, 0).fillna(0)
        out["realized_return"] = data["close"].pct_change(p.outcome_bars, fill_method=None)
        out["mature_direction"] = out["event_direction"].shift(p.outcome_bars)
        out["mature_weekday"] = out["weekday"].shift(p.outcome_bars)
        out["mature_reward"] = out["mature_direction"] * out["realized_return"]
        out["seasonal_edge"] = np.nan
        out["seasonal_samples"] = 0.0
        for weekday in range(5):
            mask = (out["mature_weekday"] == weekday) & (out["mature_direction"].abs() == 1) & out["mature_reward"].notna()
            count = mask.astype(float).rolling(p.season_window, min_periods=p.season_window).sum().shift(1)
            reward = out["mature_reward"].where(mask, 0.0).rolling(p.season_window, min_periods=p.season_window).sum().shift(1)
            out[f"samples_{weekday}"] = count
            out[f"edge_{weekday}"] = reward / count.replace(0, np.nan)
            today = out["weekday"] == weekday
            out.loc[today, "seasonal_edge"] = out.loc[today, f"edge_{weekday}"]
            out.loc[today, "seasonal_samples"] = count.loc[today]
        eligible = valid & (out["seasonal_samples"] >= p.min_samples) & (out["seasonal_edge"] >= p.min_edge)
        eligible.iloc[:self.warmup_bars(p)] = False
        out["candidate"] = out["event_direction"].where(eligible, 0).fillna(0).astype(int)
        out["conviction"] = (out["seasonal_edge"] / p.min_edge) * (out["move_strength"] / p.move_atr) * (out["relative_volume"] / p.volume_threshold)
        out["entry_size"] = (0.25 * out["conviction"]).clip(0.20, 0.95).fillna(0.20)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalVolumeParams) -> SignalFrame:
        df = pd.DataFrame({"signal": np.zeros(len(data), dtype=int), "size": np.ones(len(data), dtype=float)}, index=data.index)
        side, pending, armed, age = 0, False, False, 0
        entry, stop, peak, trade_size = 0.0, 0.0, 0.0, 1.0
        raw = df["signal"].to_numpy(copy=True)
        sizes = df["size"].to_numpy(copy=True)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].fillna(0.20).to_numpy(dtype=float)
        for i in range(len(data)):
            if side:
                if pending:
                    entry = opens[i]
                    stop = -params.initial_stop_atr * atrs[i - 1]
                    peak, age, armed, pending = 0.0, 0, False, False
                age += 1
                finite = np.isfinite([entry, stop, opens[i], closes[i], highs[i], lows[i], atrs[i]]).all()
                adverse = side * ((lows[i] if side == 1 else highs[i]) - entry)
                if not finite or entry <= 0 or adverse <= stop or age >= params.max_hold:
                    side = 0
                    # Stops observed during this bar produce a next-open exit.
                    continue
                profit = side * (closes[i] - entry)
                peak = max(peak, profit)
                armed = armed or peak >= entry * params.breakeven_pct
                if armed:
                    # Signed profit coordinates ratchet upward for either side;
                    # a short's corresponding price stop ratchets downward.
                    # Close-confirmed ratchets become active on the following bar.
                    stop = max(stop, 0.0, peak - params.trail_atr * atrs[i])
            elif candidates[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                side = int(candidates[i])
                trade_size = float(np.clip(entry_sizes[i], 0.20, 0.95))
                pending = True
            raw[i] = side
            sizes[i] = trade_size if side else 1.0
        df["signal"] = raw
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-06", periods=6, freq="B")
    close = np.array([100.0, 102.0, 103.0, 101.0, 100.0, 100.0])
    opens = np.array([100.0, 100.0, 102.0, 103.0, 101.0, 100.0])
    data = pd.DataFrame({"open": opens, "high": np.maximum(opens, close) + 0.2, "low": np.minimum(opens, close) - 0.2, "close": close, "volume": 100.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "candidate": [1, 0, 0, 0, 0, 0], "entry_size": 0.6}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, SeasonalVolumeParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert result.loc[index[1], "size"] == 0.6
    mirrored = data.copy()
    mirrored["open"] = 200.0 - data["open"]
    mirrored["close"] = 200.0 - data["close"]
    mirrored["high"] = 200.0 - data["low"]
    mirrored["low"] = 200.0 - data["high"]
    indicators["candidate"] = -indicators["candidate"]
    short_result = strategy.generate_signals(mirrored, indicators, None, SeasonalVolumeParams()).data
    assert short_result["signal"].tolist() == [0, -1, -1, -1, 0, 0]
