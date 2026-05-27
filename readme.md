# 🌦️ Deep Learning Climate Intelligence Dashboard
A hybrid forecasting system using **Long Short-Term Memory (LSTM) Neural Networks** paired with **Large Language Models (LLMs)** and real-time atmospheric data from OpenWeather.

## 🚀 Features
- **Deep Learning Forecasting**: Leverages a trained TensorFlow LSTM network to capture complex, non-linear atmospheric patterns for highly accurate temperature prediction.
- **LLM Intent Extraction**: Uses Ollama (Llama 3) to parse natural language queries into structured data, cleanly resolving relative dates like "tomorrow".
- **Dynamic Weather Integration**: Fetches real-time humidity, pressure, and dew points via OpenWeather API to feed the neural network.
- **Responsive UI**: Glassmorphism-style dashboard built with Tailwind CSS.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- **Python 3.8+**
- **Ollama** (Running locally with `llama3` model)
- **OpenWeather API Key** (Get one at [openweathermap.org](https://openweathermap.org/))

### 2. Environment Configuration
Create a `.env` file in the root directory of the project and add your credentials:
```env
OPENWEATHER_API_KEY=your_api_key_here
OLLAMA_URL=http://localhost:11434/api/generate
