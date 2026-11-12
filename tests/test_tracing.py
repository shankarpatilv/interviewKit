"""Tests for optional Langfuse tracing helpers."""

from typing import Any

import pytest

from interviewkit import tracing
from interviewkit.config import Settings


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(f"INTERVIEWKIT_{name}", raising=False)


def _enabled() -> Settings:
    return _settings(
        LANGFUSE_PUBLIC_KEY="pk-lf-test",
        LANGFUSE_SECRET_KEY="sk-lf-test",
        LANGFUSE_HOST="http://127.0.0.1:9",
    )


def test_tracing_enabled_requires_both_keys() -> None:
    assert _settings().tracing_enabled is False
    assert _settings(LANGFUSE_PUBLIC_KEY="pk").tracing_enabled is False
    assert _settings(LANGFUSE_SECRET_KEY="sk").tracing_enabled is False
    assert _enabled().tracing_enabled is True
    assert _settings().langfuse_host == "http://localhost:3000"


def test_disabled_config_has_no_callbacks_and_keeps_tags() -> None:
    config = tracing.trace_config(
        _settings(), run_name="prep", tags=["prep", "amazon"], metadata={"company": "amazon"}
    )

    assert config["run_name"] == "prep"
    assert "callbacks" not in config
    assert config["tags"] == ["prep", "amazon"]
    assert config["metadata"] == {"company": "amazon"}


def test_enabled_config_adds_one_langfuse_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    from langfuse.langchain import CallbackHandler

    monkeypatch.setattr(tracing, "_clients", {})
    config = tracing.trace_config(
        _enabled(), run_name="prep", tags=["prep"], metadata={"company": "amazon"}
    )

    callbacks = config["callbacks"]
    assert isinstance(callbacks, list) and len(callbacks) == 1
    assert isinstance(callbacks[0], CallbackHandler)
    assert config["metadata"] == {"langfuse_tags": ["prep"], "company": "amazon"}


def test_flush_is_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(_: Settings) -> None:
        raise AssertionError("client must not be built when tracing is disabled")

    monkeypatch.setattr(tracing, "get_langfuse_client", _fail)
    tracing.flush_traces(_settings())


def test_flush_calls_client_flush_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakeClient:
        def flush(self) -> None:
            calls.append("flush")

    monkeypatch.setattr(tracing, "get_langfuse_client", lambda _: _FakeClient())
    tracing.flush_traces(_enabled())

    assert calls == ["flush"]


def test_trace_run_is_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(_: Settings) -> None:
        raise AssertionError("client must not be built when tracing is disabled")

    monkeypatch.setattr(tracing, "get_langfuse_client", _fail)

    with tracing.trace_run(_settings(), name="eval:case"):
        pass


def test_trace_run_opens_root_span_with_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager
    from collections.abc import Iterator

    import langfuse

    events: list[tuple[str, Any]] = []

    class _Client:
        @contextmanager
        def start_as_current_observation(self, **kwargs: Any) -> Iterator[None]:
            events.append(("span", kwargs))
            yield

    @contextmanager
    def _propagate(**kwargs: Any) -> Iterator[None]:
        events.append(("attrs", kwargs))
        yield

    monkeypatch.setattr(tracing, "get_langfuse_client", lambda _: _Client())
    monkeypatch.setattr(langfuse, "propagate_attributes", _propagate)

    with tracing.trace_run(_enabled(), name="eval:c1", tags=("eval",), metadata={"k": "5"}):
        events.append(("body", None))

    assert events == [
        ("span", {"name": "eval:c1", "as_type": "span"}),
        ("attrs", {"trace_name": "eval:c1", "tags": ["eval"], "metadata": {"k": "5"}}),
        ("body", None),
    ]


def test_tracing_disabled_when_a_key_is_blank() -> None:
    assert not _settings(LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="sk-lf-x").tracing_enabled
