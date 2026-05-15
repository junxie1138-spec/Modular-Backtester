from pathlib import Path

from factory.dedup import append_summary, read_tail


def test_append_then_read_roundtrip(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    append_summary(d, "first idea", node_id="local")
    append_summary(d, "second idea", node_id="local")
    append_summary(d, "third idea", node_id="local")
    assert read_tail(d, n=10) == ["first idea", "second idea", "third idea"]


def test_append_writes_timestamped_line(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    append_summary(d, "an idea", node_id="desk")
    raw = (d / "desk.txt").read_text(encoding="utf-8").strip()
    ts_str, sep, summary = raw.partition("\t")
    assert sep == "\t"
    assert ts_str.isdigit()
    assert summary == "an idea"


def test_read_tail_unions_shards_by_timestamp(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    d.mkdir()
    (d / "desk.txt").write_text("100\tdesk old\n300\tdesk new\n", encoding="utf-8")
    (d / "laptop.txt").write_text("200\tlaptop mid\n", encoding="utf-8")
    # Globally sorted by timestamp, oldest first.
    assert read_tail(d, n=10) == ["desk old", "laptop mid", "desk new"]


def test_read_tail_caps_at_n(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    d.mkdir()
    lines = "".join(f"{i}\tidea {i}\n" for i in range(50))
    (d / "local.txt").write_text(lines, encoding="utf-8")
    tail = read_tail(d, n=30)
    assert len(tail) == 30
    assert tail[0] == "idea 20"
    assert tail[-1] == "idea 49"


def test_read_tail_legacy_untimestamped_line_is_oldest(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    d.mkdir()
    # A pre-migration line has no tab -> treated as timestamp 0 (always oldest).
    (d / "local.txt").write_text("legacy idea no tab\n500\tnew idea\n", encoding="utf-8")
    assert read_tail(d, n=10) == ["legacy idea no tab", "new idea"]


def test_read_tail_handles_missing_dir(tmp_path: Path) -> None:
    assert read_tail(tmp_path / "does_not_exist", n=30) == []


def test_append_strips_newlines_inside_summary(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    append_summary(d, "line1\nline2\rline3", node_id="local")
    assert read_tail(d, n=10) == ["line1 line2 line3"]


def test_append_skips_empty_or_whitespace(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    append_summary(d, "", node_id="local")
    append_summary(d, "   ", node_id="local")
    append_summary(d, "real entry", node_id="local")
    assert read_tail(d, n=10) == ["real entry"]
