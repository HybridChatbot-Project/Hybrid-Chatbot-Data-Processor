from flask import Blueprint, request, jsonify
from app.services.uniquebot_service import handle_local_message
from app.services.llama_service import handle_llama_message
from app.models.router_model import router_predict
import random

chat_bp = Blueprint("chat", __name__)

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

    # Clasificar tipo de mensaje
    prediction = router_predict(message)
    print(f"🔍 Predicción del router: {prediction} | Mensaje: {message}")

    # Si es simple → responder localmente
    if prediction == "simple":
        answer = handle_local_message(message)
        if answer:
            return jsonify({
                "conversationId": conversation_id,
                "sequence": sequence + 1,
                "message": answer,
                "senderType": "bot",
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
    })
