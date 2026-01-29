# climate_airflow_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
# Add path to your script
sys.path.append('/path/to/your/scripts')
from climate_engine import generate_forecasted_dataset

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG('climate_forecast_pipeline', 
         default_args=default_args, 
         schedule_interval='@monthly') as dag:

    run_forecast = PythonOperator(
        task_id='update_predictions_csv',
        python_callable=generate_forecasted_dataset,
        op_kwargs={
            'input_csv': 'GlobalLandTemperaturesByCountry_clean.csv',
            'output_csv': 'climate_master_data.csv'
        }
    )