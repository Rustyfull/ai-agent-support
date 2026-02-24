from pydantic_settings import BaseSettings
from functools import lru_cache
#(least_recently_used_cache)

class Settings(BaseSettings):
    """
    Zentrale Konfiguration für die Anwendung.
    Werte werden aus .env Datei oder Umgebungsvariablen geladen
    """

    # Gemini API Konfiguration
    google_api_key:str
    gemini_model:str = "gemini-2.5-flash"

    # API Server Einstellungen
    api_host: str = "0.0.0.0"
    api_port:str = 8000
    debug:bool = True

    # Agent Einstellungen
    confidence_threshold: float = 0.7   #   Unter diesem Wert -> Human Handoff
    max_tokens: int = 1024
    temperature: float = 0.3    # Niedriger = deterministischer


    class Config:
        env_file:str = ".env"
        case_sensitive:bool = False


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton Pattern: Settings werden nur einmal geladen
    """
    return Settings()



