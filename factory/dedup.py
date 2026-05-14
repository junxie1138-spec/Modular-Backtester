from __future__ import annotations

from pathlib import Path


def append_summary(log_path: Path, summary: str) -> None:
    """Append one one_line_summary to the dedup log.

    Newlines/carriage returns inside the summary are replaced with spaces so
    one line == one entry. Empty/whitespace-only summaries are silently ignored.
    Parent directories are created on demand.
    """
    cleaned = " ".join(summary.replace("\r", "\n").split("\n")).strip()
    if not cleaned:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(cleaned + "\n")


def read_tail(log_path: Path, n: int) -> list[str]:
    """Return the last n non-empty lines of the dedup log, oldest first.

    Returns [] if the file does not exist.
    """
    if not log_path.exists():
        return []
    lines = [
        line.rstrip("\n")
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines[-n:]
