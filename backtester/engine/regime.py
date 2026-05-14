from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class SpyEmaGate:
    """SPY-vs-200-day-EMA gate with hysteresis.

    trip:    spy_close[i] < spy_ema[i] * (1 + trip_pct)     (trip_pct typically -0.02)
    resume:  spy_close[i] > spy_ema[i] * (1 + resume_pct)   (resume_pct typically  0.02)
    """
    ema_lookback: int = 200
    trip_pct: float = -0.02
    resume_pct: float = 0.02
    tripped: bool = False
    tripped_history: list[bool] = field(default_factory=list)

    def update(self, *, bar_idx: int, spy_close: pd.Series, spy_ema: pd.Series) -> None:
        c = float(spy_close.iloc[bar_idx])
        e = float(spy_ema.iloc[bar_idx])
        trip_value = e * (1.0 + self.trip_pct)
        resume_value = e * (1.0 + self.resume_pct)
        if not self.tripped:
            if c < trip_value:
                self.tripped = True
        else:
            if c > resume_value:
                self.tripped = False
        self.tripped_history.append(self.tripped)


@dataclass
class VixGate:
    """VIX hysteresis gate.

    trip:    last trip_consec closes > trip_threshold
    resume:  last resume_consec closes < resume_threshold
    """
    trip_threshold: float = 30.0
    trip_consec: int = 2
    resume_threshold: float = 25.0
    resume_consec: int = 3
    tripped: bool = False
    tripped_history: list[bool] = field(default_factory=list)

    def update(self, *, bar_idx: int, vix_close: pd.Series) -> None:
        if not self.tripped:
            window = vix_close.iloc[max(0, bar_idx - self.trip_consec + 1): bar_idx + 1]
            if len(window) >= self.trip_consec and (window > self.trip_threshold).all():
                self.tripped = True
        else:
            window = vix_close.iloc[max(0, bar_idx - self.resume_consec + 1): bar_idx + 1]
            if len(window) >= self.resume_consec and (window < self.resume_threshold).all():
                self.tripped = False
        self.tripped_history.append(self.tripped)


@dataclass
class CircuitBreakerGate:
    """Rolling-N-day strategy-PnL kill switch.

    trip:    rolling pnl_window_days sum / initial_cash <= trip_pct (negative)
    resume:  pause_days bars after the trip bar (PRD literal: full size on day pause_days+1).
    """
    pnl_window_days: int = 20
    trip_pct: float = -0.05
    pause_days: int = 10
    tripped: bool = False
    _trip_bar_idx: int = -1
    tripped_history: list[bool] = field(default_factory=list)

    def update(self, *, bar_idx: int, recent_pnl: pd.Series, initial_cash: float) -> None:
        if not self.tripped:
            window = recent_pnl.iloc[max(0, bar_idx - self.pnl_window_days + 1): bar_idx + 1]
            rolling_pct = float(window.sum()) / float(initial_cash) if initial_cash else 0.0
            if rolling_pct <= self.trip_pct:
                self.tripped = True
                self._trip_bar_idx = bar_idx
        else:
            # Resume on bar _trip_bar_idx + pause_days + 1 (PRD: "full size on day 11").
            if bar_idx > self._trip_bar_idx + self.pause_days:
                self.tripped = False
        self.tripped_history.append(self.tripped)


@dataclass(frozen=True)
class RegimeState:
    """End-of-bar snapshot of which gates are tripped."""
    spy_ema_tripped: bool
    vix_tripped: bool
    circuit_breaker_tripped: bool

    @property
    def book_flat(self) -> bool:
        return self.spy_ema_tripped or self.vix_tripped or self.circuit_breaker_tripped


@dataclass
class RegimePolicy:
    """Three-gate regime policy. Disabled gates are no-ops."""
    spy_ema: SpyEmaGate
    vix: VixGate
    circuit_breaker: CircuitBreakerGate
    spy_ema_enabled: bool = False
    vix_enabled: bool = False
    circuit_breaker_enabled: bool = False

    @classmethod
    def from_disabled(cls) -> "RegimePolicy":
        return cls(
            spy_ema=SpyEmaGate(),
            vix=VixGate(),
            circuit_breaker=CircuitBreakerGate(),
        )

    @classmethod
    def from_config(cls, cfg) -> "RegimePolicy":
        """Construct from a RegimesConfig dataclass."""
        return cls(
            spy_ema=SpyEmaGate(
                ema_lookback=cfg.spy_ema.ema_lookback,
                trip_pct=cfg.spy_ema.trip_pct,
                resume_pct=cfg.spy_ema.resume_pct,
            ),
            vix=VixGate(
                trip_threshold=cfg.vix.trip_threshold,
                trip_consec=cfg.vix.trip_consec,
                resume_threshold=cfg.vix.resume_threshold,
                resume_consec=cfg.vix.resume_consec,
            ),
            circuit_breaker=CircuitBreakerGate(
                pnl_window_days=cfg.circuit_breaker.pnl_window_days,
                trip_pct=cfg.circuit_breaker.trip_pct,
                pause_days=cfg.circuit_breaker.pause_days,
            ),
            spy_ema_enabled=cfg.spy_ema.enabled,
            vix_enabled=cfg.vix.enabled,
            circuit_breaker_enabled=cfg.circuit_breaker.enabled,
        )

    def update(
        self,
        *,
        bar_idx: int,
        aux_data: dict[str, pd.DataFrame],
        recent_pnl: pd.Series,
        initial_cash: float,
    ) -> None:
        if self.spy_ema_enabled and "SPY" in aux_data:
            spy_close = aux_data["SPY"]["close"]
            spy_ema = spy_close.ewm(span=self.spy_ema.ema_lookback, adjust=False).mean()
            self.spy_ema.update(bar_idx=bar_idx, spy_close=spy_close, spy_ema=spy_ema)
        if self.vix_enabled and "^VIX" in aux_data:
            self.vix.update(bar_idx=bar_idx, vix_close=aux_data["^VIX"]["close"])
        if self.circuit_breaker_enabled:
            self.circuit_breaker.update(
                bar_idx=bar_idx, recent_pnl=recent_pnl, initial_cash=initial_cash,
            )

    def state(self, *, bar_idx: int) -> RegimeState:
        return RegimeState(
            spy_ema_tripped=self.spy_ema.tripped if self.spy_ema_enabled else False,
            vix_tripped=self.vix.tripped if self.vix_enabled else False,
            circuit_breaker_tripped=self.circuit_breaker.tripped if self.circuit_breaker_enabled else False,
        )
