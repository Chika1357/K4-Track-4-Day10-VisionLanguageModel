import os
from pathlib import Path
import shutil

from lab import run


def test_assignment_cli_is_found_when_python_venv_is_not_activated(tmp_path, monkeypatch):
    binary_directory = tmp_path / "isolated_bin"
    binary_directory.mkdir()
    cli = binary_directory / ("kaggle.exe" if os.name == "nt" else "kaggle")
    cli.write_bytes(b"test fixture, not executed")
    cli.chmod(0o755)
    monkeypatch.setattr(run.sys, "executable", str(binary_directory/"python"))
    environment = {"PATH": str(tmp_path/"empty"), "TEST_MARKER": "preserved"}
    assert shutil.which("kaggle", path=environment["PATH"]) is None
    def child(command, check, env):
        assert Path(shutil.which("kaggle", path=env["PATH"])) == cli
        assert env["TEST_MARKER"] == "preserved"
    monkeypatch.setattr(run.subprocess, "run", child)
    run.run_command("kaggle", "--help", env=environment)
    assert environment["PATH"] == str(tmp_path/"empty")
