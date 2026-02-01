import json
import pickle
import requests
import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- CONFIGURATION ---
# Replace with your actual key
OPENWEATHER_API_KEY = "d4360e87e7adad8f9fb51783dfd4ec82" 
OLLAMA_URL = "http://localhost:11434/api/generate"

# --- LOAD ASSETS ---
try:
    with open('arima_model.pkl', 'rb') as f:
        model_fit = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        model_features = pickle.load(f)
    print("✅ Model and Features loaded successfully.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

# --- ROUTES ---

@app.route('/')
def index():
    """Renders the main dashboard UI."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_weather():
    """Main endpoint to handle user chat, API fetch, and ML prediction."""
    try:
        user_message = request.json.get('message')
        print(f"\n[USER]: {user_message}")

        # 1. LLM Extraction (Date-Aware)
        intent = extract_intent_with_llm(user_message)
        city = intent.get('city', 'Unknown')
        target_date = intent.get('date', 'today')
        print(f"[LLM]: City={city}, Date={target_date}")

        # 2. Get Live Features from OpenWeather
        live_data = get_live_weather(city)
        if not live_data:
            return jsonify({"error": f"Could not find weather data for '{city}'"}), 404

        # 3. Prepare Features for the ARIMA Model
        # We need to match the exact column order used during training
        exog_df = pd.DataFrame([live_data])
        
        # Identify columns model expects (everything in features.pkl except target)
        required_exog_cols = [c for c in model_features if c != 'Temp_C']
        
        # Handle multi-valued weather dummies (Fog, Snow, etc.)
        for col in required_exog_cols:
            if col not in exog_df.columns:
                exog_df[col] = 0 # Assume the weather condition is absent (0)
        
        # Reorder and filter columns
        exog_input = exog_df[required_exog_cols]

        # 4. Predict using the ARIMA pkl file
        # Using .iloc[0] to handle position-based indexing correctly
        forecast = model_fit.forecast(steps=1, exog=exog_input)
        predicted_temp = forecast.iloc[0] 

        # 5. Return JSON Response
        return jsonify({
            "status": "success",
            "city": city,
            "date": target_date,
            "input_features": live_data,
            "predicted_temp_c": round(float(predicted_temp), 2),
            "message": f"Based on the atmospheric data for {city}, the predicted temperature for {target_date} is {predicted_temp:.2f}°C."
        })

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- HELPER FUNCTIONS ---

def extract_intent_with_llm(user_input):
    """Uses Ollama to parse chat. Includes today's date so it can resolve 'tomorrow'."""
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

        # Basic validation if city is missing
        if not parsed.get("city") or parsed["city"].lower() == "unknown":
            # Simple keyword extraction as fallback
            words = user_input.split()
            if "in" in words:
                parsed["city"] = words[words.index("in") + 1].strip("?.!")
        
        return parsed
    except:
        return {"city": "Unknown", "date": "today"}

def get_live_weather(city):
    """Fetches weather data from OpenWeather and calculates derived features."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("cod") != 200: return None
        
        temp = res['main']['temp']
        rh = res['main']['humidity']
        
        # Match features to the dataset's specific column names
        return {
            "Dew Point Temp_C": temp - ((100 - rh) / 5), # Approx formula
            "Rel Hum_%": rh,
            "Wind Speed_km/h": res['wind']['speed'] * 3.6,
            "Press_kPa": res['main']['pressure'] / 10,
            "Visibility_km": res.get('visibility', 10000) / 1000
        }
    except:
        return None

if __name__ == '__main__':
    # Running on port 5000 with debug enabled for development
    app.run(debug=True, port=5000)