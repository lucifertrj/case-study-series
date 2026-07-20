from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    QDRANT_URL: str
    QDRANT_API_KEY: str
    LLM_API_KEY: str

    LLM_MODEL: str = "gpt-5.4-nano"
    COGNEE_DATASET: str = "company_kb"
    TOP_K: int = 5

    UPLOAD_DIR: str = "/tmp/rfp-uploads"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
