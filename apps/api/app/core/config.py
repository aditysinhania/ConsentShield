from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "ConsentShield"
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "change-me"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"

    DATABASE_URL: str = "postgresql+asyncpg://consentshield:consentshield@localhost:5432/consentshield"
    DATABASE_URL_SYNC: str = "postgresql://consentshield:consentshield@localhost:5432/consentshield"
    REDIS_URL: str = "redis://localhost:6379/0"

    CORS_ORIGINS: str = "http://localhost:5173"

    STORAGE_ROOT: str = "./storage"
    SCREENSHOTS_DIR: str = "./storage/screenshots"
    REPORTS_DIR: str = "./storage/reports"

    MODELS_ROOT: str = "./models"
    VISION_MODEL_PATH: str = ""
    TEXT_MODEL_PATH: str = ""
    FUSION_MODEL_PATH: str = ""
    AI_STUB_MODE: bool = True

    TEXT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    VISION_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 16
    CACHE_MODELS: bool = True
    PHASE4_BACKEND: str = "auto"

    AI_PROVIDER: str = "offline"
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    OPENAI_MODEL: str = "gpt-4o-mini"

    API_V1_PREFIX: str = "/api/v1"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
