import os
import json
import pickle
import requests
import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

load_dotenv()
app = Flask(__name__)

# --- CONFIGURATION ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# --- LOAD ASSETS ---

try:
    # Adding compile=False bypasses the 'mse' deserialization bug!
    model_fit = load_model('lstm_model.h5', compile=False)
    with open('features.pkl', 'rb') as f:
        model_features = pickle.load(f)
    print("✅ LSTM Model and Features loaded successfully.")
except Exception as e:
    print(f"❌ Initialization Error: Please run train_model.py first! Details: {e}")

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_weather():
    try:
        user_message = request.json.get('message')
        print(f"\n[USER]: {user_message}")

        # 1. LLM Extraction
        intent = extract_intent_with_llm(user_message)
        city = intent.get('city', 'Unknown')
        target_date = intent.get('date', 'today')

        # 2. Get Live Features
        live_data = get_live_weather(city)
        if not live_data:
            return jsonify({"error": f"Could not find weather data for '{city}'. Check spelling."}), 404

        # 3. Prepare Features
        exog_df = pd.DataFrame([live_data])
        required_exog_cols = [c for c in model_features if c != 'Temp_C']
        
        for col in required_exog_cols:
            if col not in exog_df.columns:
                exog_df[col] = 0 
        
        exog_input = exog_df[required_exog_cols]

        # 4. Predict
        lstm_input = np.array(exog_input).reshape((1, 1, exog_input.shape[1]))
        forecast = model_fit.predict(lstm_input, verbose=0)
        predicted_temp = float(forecast[0][0]) 

        # 5. Return Response
        return jsonify({
            "status": "success",
            "city": city,
            "date": target_date,
            "input_features": live_data,
            "predicted_temp_c": round(predicted_temp, 2),
            "message": f"Based on the atmospheric data for {city}, the neural network predicted a temperature of {predicted_temp:.2f}°C for {target_date}."
        })

    except Exception as e:
        print("\n--- FULL ERROR TRACEBACK ---")
        traceback.print_exc()
        return jsonify({"error": "Model prediction failed. Check terminal for shape mismatch."}), 500

# --- HELPER FUNCTIONS ---
def extract_intent_with_llm(user_input):
    today_str = datetime.date.today().strftime("%B %d, %Y")
    prompt = f"""
    Today is {today_str}. 
    Task: Extract 'city' and 'date' from the user's weather query.
    User Query: "{user_input}"
    Rules:
    - If user says 'tomorrow', calculate date relative to {today_str}.
    - Format response ONLY as JSON: {{"city": "CityName", "date": "YYYY-MM-DD"}}.
    - Do not include conversational text.
    """
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3:latest",
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }, timeout=15)
        res_text = response.json().get('response', '{}').strip()
        parsed = json.loads(res_text)
        if not parsed.get("city") or parsed["city"].lower() == "unknown":
            words = user_input.split()
            if "in" in words:
                parsed["city"] = words[words.index("in") + 1].strip("?.!")
        return parsed
    except:
        return {"city": "Unknown", "date": "today"}

def get_live_weather(city):
    if not OPENWEATHER_API_KEY:
        return None
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("cod") != 200: return None
        temp = res['main']['temp']
        rh = res['main']['humidity']
        return {
            "Dew Point Temp_C": temp - ((100 - rh) / 5),
            "Rel Hum_%": rh,
            "Wind Speed_km/h": res['wind']['speed'] * 3.6,
            "Press_kPa": res['main']['pressure'] / 10,
            "Visibility_km": res.get('visibility', 10000) / 1000
        }
    except:
        return None

if __name__ == '__main__':
    app.run(debug=True, port=5000)