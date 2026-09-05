"""Choosing the model is the merchant's decision, not a constant in this repo.

Three properties, and the third is the one that matters commercially:

1. **One setting, honoured everywhere.** `RECOAGENT_MODEL` is read by every
   entry point, so a merchant configures a model once rather than passing
   `--model` to four commands with three different defaults.
2. **An explicit choice still wins.** A default that cannot be overridden from
   the command line is a cage.
3. **A local model needs no API key and no network.** That is the answer to "a
   settlement book must not leave this machine" -- the deterministic
   reconciliation never calls out at all, and with Ollama, LM Studio or vLLM
   neither does the tier that explains what it could not match.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from recoagent import llm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_ambient_choice(monkeypatch):
    """The developer's own .env must not decide what these tests observe."""
    monkeypatch.delenv(llm.MODEL_ENV, raising=False)
    monkeypatch.setattr(llm, "load_env_cache", None, raising=False)


def test_the_environment_variable_chooses_the_model(monkeypatch):
    monkeypatch.setenv(llm.MODEL_ENV, "ollama/llama3.1")
    assert llm.default_model() == "ollama/llama3.1"
    # ...and beats a caller's own default, because the merchant outranks us.
    assert llm.default_model("gemini/gemini-3.5-flash-lite") == "ollama/llama3.1"


def test_without_a_setting_the_callers_own_default_survives():
    """`categorize.run` ships a different default from the rest, and the
    published C2 artifact was measured on it. If the fallback stopped being
    honoured, CI's byte-for-byte C2 replay would fail for a reason that has
    nothing to do with categorisation."""
    assert llm.default_model() == llm.FALLBACK_MODEL
    assert llm.default_model("gemini/gemini-3.5-flash-lite") == (
        "gemini/gemini-3.5-flash-lite"
    )


@pytest.mark.parametrize("provider", sorted(llm.PROVIDERS))
def test_every_advertised_provider_can_be_built(provider):
    """`describe()` is a promise. A provider it lists and `client_for` cannot
    construct is a promise the first merchant to try it discovers is false.

    Construction is lazy by design, so this reaches no network and needs no
    credential -- which is exactly why it is safe to assert over all of them.
    """
    pytest.importorskip("openai")
    chat = llm.client_for(f"{provider}/some-model")
    assert chat.label
    assert provider in llm.describe()


@pytest.mark.parametrize("provider", sorted(llm.LOCAL_PROVIDERS))
def test_a_local_model_needs_no_api_key(provider, monkeypatch):
    """The privacy claim, checked rather than asserted: building a client for a
    local host must not consult any credential. Every key variable is removed
    from the environment first, so a developer's own .env cannot make this pass.
    """
    pytest.importorskip("openai")
    for _, key_env, _ in llm.PROVIDERS.values():
        if key_env:
            monkeypatch.delenv(key_env, raising=False)
    monkeypatch.setattr(
        llm, "require_key",
        lambda *a, **k: pytest.fail("a local provider asked for an API key"),
    )
    chat = llm.client_for(f"{provider}/local-model")
    chat.check_ready()          # builds the SDK client for real
    assert llm.PROVIDERS[provider][1] is None


def test_local_providers_point_at_localhost():
    """A 'local' provider that reached a remote host would be the worst
    possible bug in this file: the merchant believes nothing left the machine."""
    for provider in llm.LOCAL_PROVIDERS:
        base_url, key_env, _ = llm.PROVIDERS[provider]
        assert "localhost" in base_url or "127.0.0.1" in base_url, provider
        assert key_env is None, provider


def test_an_unknown_provider_names_the_ones_that_exist():
    with pytest.raises(ValueError) as exc:
        llm.client_for("nosuchprovider/x")
    message = str(exc.value)
    for provider in ("ollama", "openai", "anthropic"):
        assert provider in message
    assert "openai-compatible:" in message


def test_the_check_command_fails_on_a_model_it_cannot_reach():
    """`--check` exists so a merchant finds out before a long run, which means
    it must report failure as failure. It reported a dead local server as OK
    until it started reading the reply's error field: `send` returns transport
    problems in the reply rather than raising, so a check that only caught
    exceptions saw nothing wrong."""
    out = subprocess.run(
        [sys.executable, "-m", "recoagent.llm", "--check",
         "--model", "nosuchprovider/x"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode != 0
    assert "FAILED" in out.stdout


def test_describe_runs_without_a_model_configured():
    text = llm.describe()
    assert llm.MODEL_ENV in text
    assert "openai-compatible:" in text
    # The deterministic core needs no model, and the listing has to say so or
    # a reader concludes an API key is a prerequisite for the whole project.
    assert "no model at all" in text
