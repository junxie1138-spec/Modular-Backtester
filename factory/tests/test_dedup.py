from pathlib import Path

from factory.dedup import append_summary, read_tail


def test_append_then_read_roundtrip(tmp_path: Path) -> None:
    log = tmp_path / "dedup.txt"
    append_summary(log, "first idea")
    append_summary(log, "second idea")
    append_summary(log, "third idea")
    assert read_tail(log, n=10) == ["first idea", "second idea", "third idea"]


def test_read_tail_caps_at_n(tmp_path: Path) -> None:
    log = tmp_path / "dedup.txt"
    for i in range(50):
        append_summary(log, f"idea {i}")
    tail = read_tail(log, n=30)
    assert len(tail) == 30
    assert tail[0] == "idea 20"
    assert tail[-1] == "idea 49"


def test_read_tail_handles_missing_file(tmp_path: Path) -> None:
    assert read_tail(tmp_path / "does_not_exist.txt", n=30) == []


def test_append_creates_parent_dir(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "dir" / "dedup.txt"
    append_summary(log, "hello")
    assert log.exists()
    assert log.read_text(encoding="utf-8").strip() == "hello"


def test_append_strips_newlines_inside_summary(tmp_path: Path) -> None:
    log = tmp_path / "dedup.txt"
    append_summary(log, "line1\nline2\rline3")
    assert read_tail(log, n=10) == ["line1 line2 line3"]


def test_append_skips_empty_or_whitespace(tmp_path: Path) -> None:
    log = tmp_path / "dedup.txt"
    append_summary(log, "")
    append_summary(log, "   ")
    append_summary(log, "real entry")
    assert read_tail(log, n=10) == ["real entry"]
