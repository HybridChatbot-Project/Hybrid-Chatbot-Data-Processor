from flask import Blueprint, request, jsonify
from app.services.uniquebot_service import handle_local_message
from app.services.llama_service import handle_llama_message
from app.models.router_model import router_predict
import random

chat_bp = Blueprint("chat", __name__)

FEEDBACK_PENDING = {}
NEGATIVE_FEEDBACK = {
    "no",
    "no me ayudó",
    "no me ayudo",
    "no me sirvio",
    "no me sirvió",
    "no sirvio",
    "sigue igual",
    "no funciona",
    "necesito mas",
    "necesito ayuda"
}
POSITIVE_FEEDBACK = {
    "si",
    "sí",
    "gracias",
    "me sirvio",
    "me sirvió",
    "funciono",
    "funcionó",
    "listo",
    "todo bien"
}

@chat_bp.route("/process-message", methods=["POST"])
def process_message():
    data = request.get_json()
    message = data.get("message", "")
    conversation_id = data.get("conversationId")
    sequence = data.get("sequence")
    user_name = data.get("userName")

    # Validación mínima
    if not message or not conversation_id:
        return jsonify({"error": "Datos inválidos"}), 400

    normalized = message.lower().strip()

    if FEEDBACK_PENDING.get(conversation_id):
        FEEDBACK_PENDING[conversation_id] = False
        if any(term in normalized for term in NEGATIVE_FEEDBACK):
            answer = handle_llama_message(
                message="El usuario indicó que la respuesta local no le ayudó: " + message,
                user_name=user_name,
                conversation_id=conversation_id
            )
            return jsonify({
                "conversationId": conversation_id,
                "sequence": sequence + 1,
                "message": answer,
                "senderType": "bot",
                "source": "llm"
            })
        if any(term in normalized for term in POSITIVE_FEEDBACK):
            return jsonify({
                "conversationId": conversation_id,
                "sequence": sequence + 1,
                "message": "Gracias por confirmarlo. Si necesitas algo más, aquí estoy.",
                "senderType": "bot",
                "source": "local"
            })
        # En cualquier otro caso seguimos con el flujo normal.

    # Clasificar tipo de mensaje
    prediction = router_predict(message)
    print(f"🔍 Predicción del router: {prediction} | Mensaje: {message}")

    # Si es simple → responder localmente
    if prediction == "simple":
        answer = handle_local_message(message)
        if answer:
            FEEDBACK_PENDING[conversation_id] = True
            feedback_prompt = " ¿Te sirvió esta respuesta? Respóndeme 'sí' si fue útil o 'no' para que busque más ayuda."
            return jsonify({
                "conversationId": conversation_id,
                "sequence": sequence + 1,
                "message": f"{answer} {feedback_prompt}",
                "senderType": "bot",
                "source": "local"
            })
        else:
            print("ℹ️ Router clasificó como simple, pero no se encontró una respuesta local. Escalando a Llama.")

    # Caso contrario → enviar al modelo Llama 3.2
    answer = handle_llama_message(
        message=message,
        user_name=user_name,
        conversation_id=conversation_id
    )
    return jsonify({
        "conversationId": conversation_id,
        "sequence": sequence + 1,
        "message": answer,
        "senderType": "bot",
        "source": "llm"
    })
