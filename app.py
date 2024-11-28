from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import openai
from openai import ChatCompletion  # Asume que tienes el paquete openai instalado
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.preprocessing import LabelEncoder
import json
import io

# Cargar el tokenizador
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Cargar el LabelEncoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Cargar configuración
with open("config.json", "r") as f:
    config = json.load(f)
max_length = config["max_length"]

# Configura tu clave API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")


# Cargar modelos
modelo_texto = tf.keras.models.load_model("final_model_emotions-2.h5")
modelo_imagen = tf.keras.models.load_model("emotion_58acc_model.h5")

# Diccionario de emociones (actualiza según los modelos)
labels = ['happy', 'sad', 'angry', 'fear', 'love', 'surprise']
emotion_mapping = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}

labels = label_encoder.fit_transform(labels)
# Inicializar la aplicación Flask
app = Flask(__name__)

def procesar_imagen(imagen_base64):
    """Convierte la imagen en base64 a un formato procesable por el modelo."""
    imagen_decodificada = base64.b64decode(imagen_base64)
    imagen = Image.open(BytesIO(imagen_decodificada)).convert("L")  # Convertir a escala de grises
    imagen = imagen.resize((48, 48))  # Cambia al tamaño esperado por tu modelo
    imagen_array = np.array(imagen) / 255.0  # Normalizar
    imagen_array = np.expand_dims(imagen_array, axis=-1)  # Añadir canal
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir batch
    return imagen_array


def generar_recomendacion(emocion_texto, emocion_imagen):
    """Usa la API de ChatGPT para generar recomendaciones."""
    prompt = f"""
    Las emociones detectadas son:
    - Texto: {emocion_texto}
    - Imagen: {emocion_imagen}

    Por favor, con base en estas emociones, recomienda contenido como una canción, un video, 
    una frase motivacional, o acciones que puedan ayudar a mejorar el estado de ánimo de la persona.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def predict_emotion(text):
    """
    Función para predecir la emoción basada en texto.
    """
    try:
        # Tokenizar y preprocesar el texto
        input_sequence = tokenizer.texts_to_sequences([text])
        padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)

        # Realizar la predicción
        prediction = modelo_texto.predict(padded_input_sequence)

        # Obtener el índice de la predicción más probable
        predicted_index = np.argmax(prediction)

        # Decodificar la etiqueta utilizando el label_encoder
        predicted_label = label_encoder.inverse_transform([predicted_index])
        print(predicted_label)

        return predicted_label[0]
    except Exception as e:
        raise ValueError(f"Error en la predicción de emoción: {str(e)}")


@app.route("/analizar", methods=["POST"])
def analizar():
    """Endpoint para analizar emociones en texto e imagen."""
    datos = request.json
    texto = datos.get("texto", "")
    imagen_base64 = datos.get("imagen", "")
    
    if not texto or not imagen_base64:
        return jsonify({"error": "Debe proporcionar texto e imagen en base64"}), 400

    try:
        # Preprocesar el texto
        emocion_texto = predict_emotion(texto)

        if 'imagen' not in datos:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decodificar la imagen de base64
        image_base64 = datos['imagen']
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('L')  # Convertir a escala de grises
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Redimensionar mientras preserva la relación de aspecto
        image.thumbnail((48, 48))  # Redimensionar manteniendo proporción
        new_image = Image.new("L", (48, 48), "white")  # Imagen blanca 48x48
        new_image.paste(image, ((48 - image.size[0]) // 2, (48 - image.size[1]) // 2))  # Centrar

        # Convertir a array y normalizar
        image_array = np.array(new_image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Añadir dimensión para batch

        # Hacer la predicción
        predictions = modelo_imagen.predict(image_array)
        emotion_index = np.argmax(predictions)
        predicted_emotion = emotion_mapping[emotion_index]
        print(type(emocion_texto))

        recomendaciones = generar_recomendacion(emocion_texto, predicted_emotion)

        return jsonify({
            "emocion_texto": emocion_texto,
            "emocion_imagen": predicted_emotion,  # Descomenta si procesas la imagen
            "recomendaciones": recomendaciones  # Descomenta si generas recomendaciones
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)