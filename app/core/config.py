 
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
      API_V1_STR:str ="/api/v1"
      PROJECT_NAME: str = "Document Q&A API"
      UPLOAD_DIRECTORY: str = "uploads"
      MODEL_NAME: str = "meta-llama/Llama-3.2-1B"
      SENTENCE_EMBEDDING_NAME: str = "all-MiniLM-L6-v2"

      class Config:
         env_file = ".env"
settings = Settings()
      