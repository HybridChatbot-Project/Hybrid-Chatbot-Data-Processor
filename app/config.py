import os

SECRET_KEY = os.getenv("JWT_SECRET")
LLAMA_API_KEY = os.getenv("LLAMA_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY no está configurada")

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY no está configurada")

class Config:
    SECRET_KEY = os.getenv("JWT_SECRET")
    MODEL_ID = "Llama3.2-3B-Instruct:int4-qlora-eo8"
    MODEL_PATH = os.path.expanduser(f"~/.llama/models/{MODEL_ID}")      
    USE_LOCAL_THRESHOLD = 0.75 