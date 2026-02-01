# 🌦️ AI Climate Intelligence Dashboard
A hybrid forecasting system combining **ARIMA/SARIMAX** statistical models with **Large Language Models (LLMs)** and real-time atmospheric data from OpenWeather.

## 🚀 Features
- **LLM Intent Extraction**: Uses Ollama (Llama 3) to parse natural language queries into structured data.
- **Dynamic Weather Integration**: Fetches real-time humidity, pressure, and dew points via OpenWeather API.
- **Advanced Forecasting**: Leverages a trained ARIMA model for temperature prediction based on historical trends.
- **Responsive UI**: Glassmorphism-style dashboard built with Tailwind CSS.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- **Python 3.8+**
- **Ollama** (Running locally with `llama3` model)
- **OpenWeather API Key** (Get one at [openweathermap.org](https://openweathermap.org/))

### 2. Virtual Environment Setup
**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
Linux / macOS:

Bash
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
⚠️ Common Environment Errors (Windows)
1. "Scripts cannot be loaded because running scripts is disabled"
If you get an error when running .\venv\Scripts\activate, PowerShell is blocking the script. Fix: Run this command in your PowerShell terminal to allow the environment to activate for this session only:

PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
2. "OSError: [WinError 10013] Address already in use"
This happens if another app (or a crashed instance of your Flask app) is already using port 5000. Fix: Find and kill the process using port 5000:

PowerShell
netstat -ano | findstr :5000
taskkill /PID <YOUR_PID_HERE> /F
🖥️ Running the Application
Ensure your .pkl files are in the root directory.

Start the Flask server:

Bash
python app.py
Open your browser and go to: http://127.0.0.1:5000

📁 Project Structure
app.py: Flask backend & LLM logic.

arima_model.pkl: Serialized forecasting model.

features.pkl: List of training features for alignment.

templates/index.html: Dashboard UI.

requirements.txt: Python dependencies.


### **Why this README helps in an interview:**
1.  **Cross-Platform Instructions**: It shows you are aware of how different developers (or servers) will run your code.
2.  **Troubleshooting Section**: Explicitly mentioning Windows-specific errors like `ExecutionPolicy` shows you have deep practical experience and can solve deployment hurdles.
3.  **Clean Architecture**: The folder structure section makes it easy for the interviewer to understand your project at a glance.



This [Flask Windows Setup Guide](https://www.youtube.com/watch?v=Z1RJmh_OqeA) provides visual context on correctly setting up your Python path and environment variables on Windows systems.