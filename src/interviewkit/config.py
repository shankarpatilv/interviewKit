"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and defaults."""

    model_config = SettingsConfigDict(env_prefix="INTERVIEWKIT_", env_file=".env")

    database_url: str = "postgresql://interviewkit:interviewkit@localhost:5432/interviewkit"
    experiences_dir: Path = Path("experiences")
    companies_dir: Path = Path("companies")
    sessions_dir: Path = Path("sessions")
    default_provider: str = "openai"
