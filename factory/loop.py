from __future__ import annotations

import logging
import logging.handlers
import random
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from factory.cycle import run_cycle
from factory.settings_loader import Settings, load_settings
from factory.sync import bootstrap, sync_pull, sync_push

log = logging.getLogger(__name__)


class _ShutdownFlag:
    """Threadsafe set-once flag used by signal handlers."""
    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()


def configure_logging(log_path: Path) -> None:
    """Configure a rotating file handler on the `factory` logger root.

    Idempotent: re-configuring removes prior handlers first.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("factory")
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    file_h = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    file_h.setFormatter(fmt)
    root.addHandler(file_h)
    # Also mirror to stderr for interactive runs.
    stream_h = logging.StreamHandler(sys.stderr)
    stream_h.setFormatter(fmt)
    root.addHandler(stream_h)


def _model_from_flags(claude_flags: tuple[str, ...]) -> str:
    """Extract the --model value from claude_flags for display.

    settings.toml passes no --model, so the factory uses the Claude Code
    default (Opus) unless an override is set in settings.local.toml.
    """
    for i, flag in enumerate(claude_flags):
        if flag == "--model" and i + 1 < len(claude_flags):
            return claude_flags[i + 1]
        if flag.startswith("--model="):
            return flag.split("=", 1)[1]
    return "(Claude Code default)"


def _install_signal_handlers(flag: _ShutdownFlag) -> None:
    def _handler(signum, frame):  # noqa: ARG001
        log.info("received signal %s; requesting graceful shutdown", signum)
        flag.set()
    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)


def run_loop(
    settings: Settings,
    *,
    rng: random.Random,
    shutdown_flag: Optional[_ShutdownFlag] = None,
    max_cycles_override: Optional[int] = None,
) -> int:
    """Run cycles in a loop. Returns the number of cycles completed.

    Stops when shutdown_flag is set OR when max_cycles (>0) is reached.
    max_cycles_override (if provided) wins over settings.loop.max_cycles.
    """
    flag = shutdown_flag or _ShutdownFlag()
    max_cycles = max_cycles_override if max_cycles_override is not None else settings.loop.max_cycles
    sleep_sec = settings.loop.inter_cycle_sleep_sec

    mode = "distributed" if settings.sync.enabled else "standalone"
    model = _model_from_flags(settings.generation.claude_flags)
    log.info(
        "factory loop starting: node=%s mode=%s model=%s",
        settings.node_id, mode, model,
    )

    try:
        bootstrap(settings)
    except Exception as exc:
        log.exception("sync bootstrap failed (continuing): %s", exc)

    completed = 0
    while not flag.is_set():
        try:
            sync_pull(settings)
        except Exception as exc:
            log.exception("sync_pull failed (continuing): %s", exc)
        try:
            outcome = run_cycle(settings, rng=rng)
            log.info("cycle %d outcome=%s id=%s",
                     completed + 1, outcome.status, outcome.strategy_id)
        except Exception as exc:
            # An unexpected exception from inside run_cycle: log and continue.
            # (run_cycle is supposed to never raise on expected failures, so
            # reaching here means a bug — but the loop must not die.)
            log.exception("unexpected exception in run_cycle: %s", exc)
        try:
            sync_push(settings)
        except Exception as exc:
            log.exception("sync_push failed (continuing): %s", exc)
        completed += 1
        if max_cycles and completed >= max_cycles:
            break
        if flag.is_set():
            break
        if sleep_sec > 0:
            # Sleep in short increments so SIGINT is responsive.
            slept = 0.0
            while slept < sleep_sec and not flag.is_set():
                time.sleep(min(0.5, sleep_sec - slept))
                slept += 0.5
    log.info("loop stopping after %d cycles", completed)
    return completed


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser("factory.loop")
    parser.add_argument(
        "--settings",
        default="factory/config/settings.toml",
        type=Path,
        help="Path to settings.toml",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for slot pulls")
    args = parser.parse_args(argv)

    s = load_settings(args.settings)
    configure_logging(s.paths.factory_log)
    flag = _ShutdownFlag()
    _install_signal_handlers(flag)
    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    run_loop(s, rng=rng, shutdown_flag=flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
