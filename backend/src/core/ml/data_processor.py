import pandas as pd
import numpy as np

class SensorDataProcessor:
    def __init__(self):
        self.required_columns = [
            'timestamp', 'pressure', 'flow_rate', 'temperature',
            'pipeline_id', 'location'
        ]
    
    def validate_dataframe(self, df):
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colunas faltantes: {missing_cols}")
        return True