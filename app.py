import streamlit as st
import pandas as pd
import ollama
from langchain_community.chat_models import ChatOllama
from langchain_experimental.agents import create_pandas_dataframe_agent

st.set_page_config(page_title="Climate Forecasting AI", layout="wide")

st.title("🌍 Climate Change Forecast Assistant")
st.write("Ask questions about historical trends or future temperature predictions.")

# 1. Load your Data (Update path to your local Windows path)
@st.cache_data
def load_data():
    # Use the dataframe you processed in your notebook
    df = pd.read_csv("./climate_master_data.csv")
    return df

df = load_data()

# 2. Initialize the Local LLM (Ollama must be running on your Windows)
llm = ChatOllama(model="llama3", temperature=0)

# 3. Create the LangChain Agent
agent = create_pandas_dataframe_agent(
    llm, 
    df, 
    verbose=True, 
    allow_dangerous_code=True
)

# 4. User Interface
user_query = st.text_input("Enter your question (e.g., 'Compare temperature trends in India vs Brazil'): ")

if user_query:
    with st.spinner("Analyzing data..."):
        try:
            response = agent.invoke(user_query)
            st.success("Analysis Complete!")
            st.write(response['output'])
            
            # Optional: Add a simple chart based on the data
            if "India" in user_query:
                india_data = df[df['Country'] == 'India'].set_index('dt')['AverageTemperature']
                st.line_chart(india_data.tail(100)) # Show last 100 months
                
        except Exception as e:
            st.error(f"Error: {e}")