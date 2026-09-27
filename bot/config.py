from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    telegram_bot_token: str

    openrouter_api_key: str
    openrouter_model: str = "google/gemini-2.5-flash"

    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "google_drive_rag"

    redis_url: str = "redis://redis:6379/0"

    allowed_chat_ids: str = ""

    top_k: int = 5
    score_threshold: float = 0.4

    max_turns: int = 6
    memory_ttl: int = 86400

    rate_limit: int = 8
    rate_window: int = 60

    max_context_chars: int = 4500
    max_question_chars: int = 1000

    num_predict: int = 800
    temperature: float = 0.2

    app_name: str = "Telegram RAG Assistant"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def allowed_chat_id_set(self) -> set[str]:
        return {
            x.strip()
            for x in self.allowed_chat_ids.split(",")
            if x.strip()
        }


settings = Settings()
