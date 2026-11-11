"""Optional Langfuse tracing for LangGraph runs and LLM calls.

Langfuse is imported lazily so the CLI stays fast and tracing stays fully off
unless both Langfuse keys are configured.
"""

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig

from interviewkit.config import Settings, get_settings

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = logging.getLogger(__name__)

_clients: dict[str, "Langfuse"] = {}


def get_langfuse_client(app_settings: Settings) -> "Langfuse":
    """Return a Langfuse client built from explicit Settings keys and host."""
    if app_settings.langfuse_public_key is None or app_settings.langfuse_secret_key is None:
        msg = "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required for tracing"
        raise ValueError(msg)
    public_key = app_settings.langfuse_public_key.get_secret_value()
    if public_key not in _clients:
        from langfuse import Langfuse

        _clients[public_key] = Langfuse(
            public_key=public_key,
            secret_key=app_settings.langfuse_secret_key.get_secret_value(),
            base_url=app_settings.langfuse_host,
        )
        logger.debug("Langfuse tracing enabled for host %s", app_settings.langfuse_host)
    return _clients[public_key]


def trace_config(
    app_settings: Settings | None = None,
    *,
    run_name: str,
    tags: Sequence[str] = (),
    metadata: Mapping[str, str] | None = None,
) -> RunnableConfig:
    """Build a LangChain RunnableConfig, adding a Langfuse handler when enabled."""
    resolved = app_settings or get_settings()
    extra: dict[str, Any] = dict(metadata or {})
    config = RunnableConfig(run_name=run_name, tags=list(tags), metadata=extra)
    if not resolved.tracing_enabled:
        return config

    from langfuse.langchain import CallbackHandler

    get_langfuse_client(resolved)
    public_key = resolved.langfuse_public_key
    handler = CallbackHandler(public_key=public_key.get_secret_value() if public_key else None)
    config["callbacks"] = [handler]
    config["metadata"] = {"langfuse_tags": list(tags), **extra}
    return config


def flush_traces(app_settings: Settings | None = None) -> None:
    """Flush pending Langfuse events so short CLI runs do not drop traces."""
    resolved = app_settings or get_settings()
    if not resolved.tracing_enabled:
        return
    get_langfuse_client(resolved).flush()


@contextmanager
def trace_run(
    app_settings: Settings | None = None,
    *,
    name: str,
    tags: Sequence[str] = (),
    metadata: Mapping[str, str] | None = None,
) -> Iterator[None]:
    """Open one root Langfuse span so graph steps and LLM calls share a trace."""
    resolved = app_settings or get_settings()
    if not resolved.tracing_enabled:
        yield
        return

    from langfuse import propagate_attributes

    client = get_langfuse_client(resolved)
    with client.start_as_current_observation(name=name, as_type="span"):
        with propagate_attributes(trace_name=name, tags=list(tags), metadata=dict(metadata or {})):
            yield
