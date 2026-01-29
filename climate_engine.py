# climate_engine.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pickle

def generate_forecasted_dataset(input_csv, output_csv):
    # Load your clean dataset
    df = pd.read_csv(input_csv)
    # Ensure columns: country, year, avg_temp
    
    countries = df['country'].unique()
    all_results = []

    for country in countries:
        country_df = df[df['country'] == country].sort_values('year')
        if len(country_df) < 10: continue # Skip if data is too thin
        
        # Train ARIMA
        model = ARIMA(country_df['avg_temp'], order=(5,1,0))
        model_fit = model.fit()
        
        # Forecast 17 years (from 2013 to 2030)
        forecast = model_fit.forecast(steps=17)
        
        # Create future rows
        future_years = range(2014, 2031)
        for year, temp in zip(future_years, forecast):
            all_results.append({'country': country, 'year': year, 'avg_temp': temp, 'type': 'forecast'})
            
    # Add historical data
    df['type'] = 'historical'
    forecast_df = pd.DataFrame(all_results)
    final_df = pd.concat([df, forecast_df], ignore_index=True)
    
    # Save for Tableau and Streamlit
    final_df.to_csv(output_csv, index=False)
    print(f"Dataset updated with forecasts: {output_csv}")

if __name__ == "__main__":
    generate_forecasted_dataset('GlobalLandTemperaturesByCountry_clean.csv', 'climate_master_data.csv')