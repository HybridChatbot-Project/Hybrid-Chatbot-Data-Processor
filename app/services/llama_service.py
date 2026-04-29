#import requests
import os
from typing import Dict, Optional
from app.config import GROQ_API_KEY
from groq import Groq

client = Groq(api_key=GROQ_API_KEY)
# MODEL_ID = "Llama-3.2-3B-Instruct:int4-qlora-eo8"
APP_NAME = "MAYI"
CONTEXT_STORE: Dict[str, Dict[str, str]] = {}
#LLAMA_ENDPOINT = os.getenv('LLAMA_SERVER_URL', 'http://localhost:8081/completion') ##"http://localhost:8081/completion"
MENTAL_HEALTH_KEYWORDS = {
    "ansiedad", "ansioso", "ansiosa", "estrés", "estres", "estresado",
    "estresada", "panic", "pánico", "panico", "ataque", "salud mental",
    "emocion", "emociones", "triste", "deprimido", "deprimida",
    "miedo", "preocupado", "preocupada", "terapia", "psicologo",
    "psicóloga", "psicóloga", "respiración", "respirar", "calma",
    "angustia", "sintoma", "síntoma", "crisis", "autolesión", "autolesion","consejos",
    "mareo", "mareos", "vértigo", "vertigo", "nausea", "náusea"
}

def handle_llama_message(message: str, user_name: Optional[str] = None, conversation_id: Optional[str] = None) -> str:
    try:
        context = _update_context(conversation_id, user_name)
        messages = _build_messages(message, context)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.6,
            max_tokens=120
        )
        answer = completion.choices[0].message.content
        _append_history(context, message or "", answer)
        return answer
    except Exception as e:
        print(f"❌ Error al contactar con llama-server: {e}")
        return "Error al procesar el mensaje con el modelo IA."

def _update_context(conversation_id: Optional[str], user_name: Optional[str]) -> Dict[str, str]:
    key = conversation_id or "_default"
    ctx = CONTEXT_STORE.setdefault(key, {"user_name": "usuario"})
    if user_name:
        ctx["user_name"] = user_name.strip() or ctx["user_name"]
    ctx["app_name"] = APP_NAME
    ctx["conversation_id"] = key
    ctx.setdefault("last_in_scope", False)
    ctx.setdefault("history", [])  # list of tuples (user, bot)
    return ctx

def _build_messages(message: str, context: Dict[str, str]):
    user = context.get("user_name", "usuario")
    app = context.get("app_name", APP_NAME)

    messages = [
        {
            "role": "system",
            "content": (
                f"Eres {app}, un chatbot de apoyo emocional. "
                "Responde con empatía, breve (máx 60 palabras) "
                "y solo sobre salud mental."
            )
        }
    ]

    if context.get("history"):
        for u, b in context["history"][-6:]:
            if u:
                messages.append({"role": "user", "content": u})
            if b:
                messages.append({"role": "assistant", "content": b})

    messages.append({"role": "user", "content": message})
    return messages

def _is_in_scope(message: str) -> bool:
    text = (message or "").lower()
    return any(keyword in text for keyword in MENTAL_HEALTH_KEYWORDS)

def _append_history(context: Dict[str, str], user_msg: str, bot_msg: str):
    history = context.setdefault("history", [])
    history.append((user_msg, bot_msg))
    # Limitar el histroial para no crecer indefinidamente
    if len(history) > 10:
        del history[:-10]
