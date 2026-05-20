import json
import subprocess
from types import SimpleNamespace
from unittest import mock

from factory.scripts.preflight import PASS, _check_generation_provider


def test_check_generation_provider_accepts_codex_plain_stdout() -> None:
    settings = SimpleNamespace(
        generation=SimpleNamespace(provider="codex", cmd="codex", flags=("exec", "-"))
    )
    proc = subprocess.CompletedProcess(["codex", "exec", "-"], 0, stdout="ok\n", stderr="")
    with mock.patch("factory.scripts.preflight.shutil.which", return_value="/bin/codex"), \
         mock.patch("factory.scripts.preflight.subprocess.run", return_value=proc) as run:
        status, detail = _check_generation_provider(settings, probe=True)
    assert status == PASS
    assert "codex CLI authenticated" in detail
    assert run.call_args.args[0] == ["/bin/codex", "exec", "-"]


def test_check_generation_provider_keeps_claude_envelope_probe() -> None:
    settings = SimpleNamespace(
        generation=SimpleNamespace(provider="claude", cmd="claude", flags=("-p", "--output-format", "json"))
    )
    proc = subprocess.CompletedProcess(
        ["claude", "-p"], 0, stdout=json.dumps({"result": "ok"}), stderr=""
    )
    with mock.patch("factory.scripts.preflight.shutil.which", return_value="/bin/claude"), \
         mock.patch("factory.scripts.preflight.subprocess.run", return_value=proc):
        status, detail = _check_generation_provider(settings, probe=True)
    assert status == PASS
    assert "claude CLI authenticated" in detail
