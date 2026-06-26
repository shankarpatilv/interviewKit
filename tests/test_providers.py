from dataclasses import dataclass
import asyncio

import pytest

from interviewkit.config import Settings
from interviewkit.providers import factory
from interviewkit.providers import ProviderAPIError, ProviderConfigurationError, get_provider
from interviewkit.providers.claude import ClaudeProvider
from interviewkit.providers.openai import OpenAIProvider


@dataclass
class _OpenAIMessage:
    content: str


@dataclass
class _OpenAIChoice:
    message: _OpenAIMessage


@dataclass
class _OpenAIResponse:
    choices: list[_OpenAIChoice]


class _FakeOpenAICompletions:
    def __init__(self, should_fail: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.should_fail = should_fail

    async def create(self, **kwargs: object) -> _OpenAIResponse:
        self.calls.append(kwargs)
        if self.should_fail:
            raise RuntimeError("openai failed")
        return _OpenAIResponse(choices=[_OpenAIChoice(message=_OpenAIMessage(content="hello"))])


class _FakeOpenAIClient:
    def __init__(self, completions: _FakeOpenAICompletions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


@dataclass
class _ClaudeBlock:
    type: str
    text: str


@dataclass
class _ClaudeResponse:
    content: list[_ClaudeBlock]


class _FakeClaudeMessages:
    def __init__(self, should_fail: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.should_fail = should_fail

    async def create(self, **kwargs: object) -> _ClaudeResponse:
        self.calls.append(kwargs)
        if self.should_fail:
            raise RuntimeError("claude failed")
        return _ClaudeResponse(content=[_ClaudeBlock(type="text", text="hi")])


class _FakeClaudeClient:
    def __init__(self, messages: _FakeClaudeMessages) -> None:
        self.messages = messages


def test_get_provider_selects_openai_from_config() -> None:
    settings = Settings(default_provider="openai", openai_api_key="sk", _env_file=None)

    provider = get_provider(app_settings=settings)

    assert isinstance(provider, OpenAIProvider)


def test_get_provider_selects_claude_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClaudeProvider:
        def __init__(self, app_settings: Settings) -> None:
            self.app_settings = app_settings

    monkeypatch.setattr(factory, "ClaudeProvider", FakeClaudeProvider)
    settings = Settings(default_provider="claude", anthropic_api_key="ak", _env_file=None)

    provider = get_provider(app_settings=settings)

    assert isinstance(provider, FakeClaudeProvider)


def test_openai_provider_returns_normalized_response() -> None:
    async def run_test() -> None:
        completions = _FakeOpenAICompletions()
        provider = OpenAIProvider(
            app_settings=Settings(openai_api_key="sk-test", _env_file=None),
            client=_FakeOpenAIClient(completions),
        )

        response = await provider.complete([{"role": "user", "content": "hello"}])

        assert response.content == "hello"
        assert response.provider == "openai"
        assert response.model == "gpt-4o"
        assert completions.calls[0]["messages"] == [{"role": "user", "content": "hello"}]
        assert "tools" not in completions.calls[0]

    asyncio.run(run_test())


def test_claude_provider_returns_normalized_response() -> None:
    async def run_test() -> None:
        messages = _FakeClaudeMessages()
        provider = ClaudeProvider(
            app_settings=Settings(anthropic_api_key="ak-test", _env_file=None),
            client=_FakeClaudeClient(messages),
        )

        response = await provider.complete([{"role": "user", "content": "hello"}])

        assert response.content == "hi"
        assert response.provider == "claude"
        assert response.model == "claude-3-5-sonnet-20241022"
        assert messages.calls[0]["messages"] == [{"role": "user", "content": "hello"}]
        assert "tools" not in messages.calls[0]

    asyncio.run(run_test())


def test_provider_errors_are_wrapped() -> None:
    async def run_test() -> None:
        provider = ClaudeProvider(
            app_settings=Settings(anthropic_api_key="ak-test", _env_file=None),
            client=_FakeClaudeClient(_FakeClaudeMessages(should_fail=True)),
        )

        with pytest.raises(ProviderAPIError, match="Claude API call failed"):
            await provider.complete([{"role": "user", "content": "hello"}])

    asyncio.run(run_test())


def test_openai_provider_errors_are_wrapped() -> None:
    async def run_test() -> None:
        provider = OpenAIProvider(
            app_settings=Settings(openai_api_key="sk-test", _env_file=None),
            client=_FakeOpenAIClient(_FakeOpenAICompletions(should_fail=True)),
        )

        with pytest.raises(ProviderAPIError, match="OpenAI API call failed"):
            await provider.complete([{"role": "user", "content": "hello"}])

    asyncio.run(run_test())


def test_claude_provider_requires_anthropic_sdk_when_client_is_not_injected() -> None:
    settings = Settings(anthropic_api_key="ak-test", _env_file=None)

    with pytest.raises(ProviderConfigurationError, match="Anthropic SDK"):
        ClaudeProvider(app_settings=settings)
