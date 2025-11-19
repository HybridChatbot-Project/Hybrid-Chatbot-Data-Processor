import requests
from typing import Dict, Optional

MODEL_ID = "Llama-3.2-3B-Instruct:int4-qlora-eo8"
APP_NAME = "MAYI"
CONTEXT_STORE: Dict[str, Dict[str, str]] = {}
LLAMA_ENDPOINT = "http://localhost:8081/completion"
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
        in_scope = _is_in_scope(message)
        context["last_in_scope"] = context.get("last_in_scope", False) or in_scope

        prompt = _build_prompt(message, context)
        payload = {
            "prompt": prompt,
            "temperature": 0.6,
            "top_p": 0.85,
            "n_predict": 256,
        }

        response = requests.post(LLAMA_ENDPOINT, json=payload)
        if response.status_code == 200:
            data = response.json()
            answer = data.get("content") or data.get("completion") or "Sin respuesta del modelo."
            _append_history(context, message or "", answer)
            return answer
        else:
            return f"Error del servidor Llama ({response.status_code})"
    except Exception as e:
        print(f"❌ Error al contactar con llama-server: {e}")
        return "Error al procesar el mensaje con el modelo local Llama."

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

def _build_prompt(message: str, context: Dict[str, str]) -> str:
    user = context.get("user_name", "usuario")
    app = context.get("app_name", APP_NAME)
    system_instructions = (
        f"Eres {app}, un chatbot híbrido que apoya a personas con ansiedad y otras emociones afines. "
        "Responde solo temas de salud mental y bienestar emocional; declina cualquier otro tema con amabilidad. "
        "Usa un tono empático, ofrece pasos concretos y refuerza que el usuario no está solo. "
        "Mantén el hilo de la conversación: no reinicies el tema ni preguntes de nuevo lo ya dicho; responde al último mensaje en contexto. "
        "No inventes diálogos del usuario; escribe únicamente tu respuesta. "
        "Si detectas riesgo, anima a contactar líneas de ayuda locales."
    )
    history = ""
    if context.get("history"):
        turns = []
        for u, b in context["history"][-6:]:  # últimos 6 pares
            if u:
                turns.append(f"{user}: {u}")
            if b:
                turns.append(f"{app}: {b}")
        history = "Contexto previo (reciente):\n" + "\n".join(turns) + "\n\n"
    return (
        f"{system_instructions}\n\n"
        f"Datos conocidos:\n"
        f"- Nombre del usuario: {user}\n"
        f"- Conversación ID: {context.get('conversation_id', '')}\n"
        f"- Aplicación: {app}\n\n"
        f"{history}"
        f"{user}: {message}\n"
        f"{app}:"
    )

def _is_in_scope(message: str) -> bool:
    text = (message or "").lower()
    return any(keyword in text for keyword in MENTAL_HEALTH_KEYWORDS)

def _append_history(context: Dict[str, str], user_msg: str, bot_msg: str):
    history = context.setdefault("history", [])
    history.append((user_msg, bot_msg))
    # Limitar el histroial para no crecer indefinidamente
    if len(history) > 10:
        del history[:-10]
