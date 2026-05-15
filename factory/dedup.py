from __future__ import annotations

import time
from pathlib import Path


def append_summary(dedup_dir: Path, summary: str, *, node_id: str) -> None:
    """Append one timestamped one_line_summary to this machine's dedup shard.

    The shard is `dedup_dir/<node_id>.txt`; each line is `<unix-int>\t<summary>`.
    Newlines/carriage returns inside the summary are replaced with spaces so
    one line == one entry. Empty/whitespace-only summaries are silently ignored.
    Parent directories are created on demand.
    """
    cleaned = " ".join(summary.replace("\r", "\n").split("\n")).strip()
    if not cleaned:
        return
    shard = dedup_dir / f"{node_id}.txt"
    dedup_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    with shard.open("a", encoding="utf-8") as f:
        f.write(f"{ts}\t{cleaned}\n")


def read_tail(dedup_dir: Path, n: int) -> list[str]:
    """Return the globally most-recent `n` summaries across all shards, oldest first.

    Reads every `*.txt` shard in `dedup_dir`, parses each line as
    `<timestamp>\t<summary>`, sorts all entries by timestamp ascending, and
    returns the summaries of the last `n`.

    Legacy tolerance: a line with no tab (a pre-migration entry) is treated as
    timestamp 0, i.e. always oldest. Returns [] if the directory does not exist
    or if n <= 0.
    """
    if n <= 0 or not dedup_dir.exists():
        return []
    entries: list[tuple[int, str]] = []
    for shard in sorted(dedup_dir.glob("*.txt")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ts_str, sep, rest = line.partition("\t")
            if sep:
                try:
                    ts = int(ts_str)
                    summary = rest
                except ValueError:
                    ts, summary = 0, line
            else:
                ts, summary = 0, line
            entries.append((ts, summary))
    entries.sort(key=lambda e: e[0])
    return [summary for _, summary in entries[-n:]]
