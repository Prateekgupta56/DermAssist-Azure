import os
import requests
import numpy as np
import tensorflow.lite as tflite
from flask import Flask, render_template, request, send_file
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speechsdk

# --- PERMANENT PATH SETUP ---
BASE_DIR = '/content/drive/MyDrive/DermAssist_Final_Submission'
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# --- CREDENTIALS ---
# --- IN YOUR GITHUB VERSION OF app.py ---
import os

# Instead of hard-coded strings, we tell the app to look at the system settings
VISION_KEY = os.getenv("VISION_KEY", "YOUR_KEY_HERE")
VISION_URL = os.getenv("VISION_URL", "YOUR_URL_HERE")
TR_KEY = os.getenv("TR_KEY", "YOUR_KEY_HERE")
SP_KEY = os.getenv("SP_KEY", "YOUR_KEY_HERE")

# --- FILE PATHS ---
MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
AUDIO_OUTPUT = os.path.join(BASE_DIR, "advice.wav")

# Load TFLite Model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
labels = open(LABELS_PATH).read().splitlines()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")
        
    file = request.files['file']
    img_bytes = file.read()

    # 1. Local TFLite Prediction
    prediction = labels[0] # Using top label for demo
    
    advice_map = {
        "Eczema": "This condition often involves itchy, inflamed skin. Use fragrance-free moisturizers and avoid harsh soaps.",
        "Melanoma": "Warning: High-risk lesion detected. Please consult a dermatologist urgently.",
        "Nevus": "Common mole detected. Monitor for changes and see a doctor if it grows."
    }
    advice = advice_map.get(prediction, "Please consult a medical professional for a formal diagnosis.")

    # 2. Azure Translation (Hindi)
    translated_text = prediction
    try:
        tr_client = TextTranslationClient(credential=AzureKeyCredential(TR_KEY), endpoint=TR_ENDPOINT, region=REGION)
        translated_resp = tr_client.translate(body=[prediction], to_language=["hi"])
        translated_text = translated_resp[0].translations[0].text
    except Exception as e:
        print(f"Translator Error: {e}")

    # 3. Azure Speech (Audio Advice)
    audio_ready = False
    try:
        speech_config = speechsdk.SpeechConfig(subscription=SP_KEY, region=REGION)
        speech_config.speech_synthesis_voice_name = "hi-IN-SwaraNeural"
        audio_config = speechsdk.audio.AudioOutputConfig(filename=AUDIO_OUTPUT)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        # We synthesize the translated text (Hindi)
        result = synthesizer.speak_text_async(translated_text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_ready = True
    except Exception as e:
        print(f"Speech Error: {e}")

    return render_template(
        'index.html', 
        prediction=prediction, 
        advice=advice, 
        translated=translated_text, 
        audio_ready=audio_ready
    )

@app.route('/get_audio')
def get_audio():
    return send_file(AUDIO_OUTPUT, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(port=5000)