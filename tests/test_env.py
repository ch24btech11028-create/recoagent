"""The .env loader: the shell wins, and a tracked file is refused."""

from __future__ import annotations

import os

import pytest

from recoagent.env import _parse, load_env, require_key


def test_parses_the_boring_subset():
    got = _parse(
        "# a comment\n"
        "\n"
        "export A=1\n"
        'B="quoted"\n'
        "C='single'\n"
        "D=has=equals\n"
        "  E = spaced  \n"
        "NOTAKEYVALUE\n"
    )
    assert got == {"A": "1", "B": "quoted", "C": "single", "D": "has=equals", "E": "spaced"}


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "nope.env") == {}


def test_the_shell_wins(tmp_path, monkeypatch):
    """A stale .env must never shadow a variable the caller exported."""
    p = tmp_path / ".env"
    p.write_text("RECOAGENT_TEST_KEY=from_file\n")
    monkeypatch.setenv("RECOAGENT_TEST_KEY", "from_shell")
    load_env(p)
    assert os.environ["RECOAGENT_TEST_KEY"] == "from_shell"
    assert require_key("RECOAGENT_TEST_KEY", dotenv=p) == "from_shell"


def test_override_is_explicit(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("RECOAGENT_TEST_KEY=from_file\n")
    monkeypatch.setenv("RECOAGENT_TEST_KEY", "from_shell")
    load_env(p, override=True)
    assert os.environ["RECOAGENT_TEST_KEY"] == "from_file"


def test_falls_back_to_the_file(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("RECOAGENT_TEST_KEY=from_file\n")
    monkeypatch.delenv("RECOAGENT_TEST_KEY", raising=False)
    assert require_key("RECOAGENT_TEST_KEY", dotenv=p) == "from_file"


def test_missing_key_explains_both_routes(tmp_path, monkeypatch):
    monkeypatch.delenv("RECOAGENT_ABSENT_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        require_key("RECOAGENT_ABSENT_KEY", dotenv=tmp_path / "nope.env")
    msg = str(exc.value)
    assert "export RECOAGENT_ABSENT_KEY" in msg
    assert ".env" in msg


def test_refuses_a_git_tracked_env_file(tmp_path, monkeypatch):
    """A key in a tracked file is a published key. Reading it would launder that.

    `_is_tracked` is stubbed rather than driven through a real index: the point
    under test is the refusal, and asserting it against this repo's own git
    state would make the test pass or fail on whether a file happened to be
    staged.
    """
    monkeypatch.delenv("RECOAGENT_ABSENT_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text("RECOAGENT_ABSENT_KEY=leaked\n")
    monkeypatch.setattr("recoagent.env._is_tracked", lambda _p: True)
    with pytest.raises(RuntimeError, match="tracked by git"):
        require_key("RECOAGENT_ABSENT_KEY", dotenv=p)


def test_reads_an_untracked_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("RECOAGENT_ABSENT_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text("RECOAGENT_ABSENT_KEY=fine\n")
    monkeypatch.setattr("recoagent.env._is_tracked", lambda _p: False)
    assert require_key("RECOAGENT_ABSENT_KEY", dotenv=p) == "fine"
