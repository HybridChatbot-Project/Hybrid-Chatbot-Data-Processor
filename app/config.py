import os

SECRET_KEY = os.getenv("JWT_SECRET", "SecurityDataProcessedChatbotKeyJwt1")
LLAMA_API_KEY = os.getenv("LLAMA_KEY", "")

class Config:
    SECRET_KEY = os.getenv("JWT_SECRET", "SecurityDataProcessedChatbotKeyJwt1")
    MODEL_ID = "Llama3.2-3B-Instruct:int4-qlora-eo8"
    MODEL_PATH = os.path.expanduser(f"~/.llama/models/{MODEL_ID}")
    USE_LOCAL_THRESHOLD = 0.75 