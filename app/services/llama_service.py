import requests

MODEL_ID = "Llama-3.2-3B-Instruct:int4-qlora-eo8"

def handle_llama_message(message: str) -> str:
    try:
        payload = {
            "prompt": f"Eres un asistente técnico amable y profesional que ayuda a usuarios sobre el chatbot híbrido. Por otro lado, la respuesta brindamelo en un maximo de 150 palabras. \nUsuario: {message}\nAsistente:",
            "temperature": 0.7,
            "top_p": 0.9,
            "n_predict": 256,
        }

        response = requests.post("http://localhost:8081/completion", json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("content") or data.get("completion") or "Sin respuesta del modelo."
        else:
            return f"Error del servidor Llama ({response.status_code})"
    except Exception as e:
        print(f"❌ Error al contactar con llama-server: {e}")
        return "Error al procesar el mensaje con el modelo local Llama."