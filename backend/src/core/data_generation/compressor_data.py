import numpy as np
import pandas as pd
import json
from pathlib import Path

class CompressorDataGenerator:
    def __init__(self, config_path=f"{Path(__file__).parent.parent.parent.parent}\\data\\config\\generation_params.json"):
        self.load_config(config_path)
    
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            self.params = json.load(f)
    
    def generate_compressor_data(self, n_samples=5000, compressor_id="comp_001", save_path=None):
        """Gera dados sintéticos para compressores"""
        comp_params = self.params['compressor_params']
        
        data = {
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='min'),
            'compressor_id': [compressor_id] * n_samples,
            'bearing_temperature': np.random.normal(
                comp_params['bearing_temperature']['mean'],
                comp_params['bearing_temperature']['std'],
                n_samples
            ),
            'oil_pressure': np.random.normal(
                comp_params['oil_pressure']['mean'],
                comp_params['oil_pressure']['std'],
                n_samples
            ),
            'power_consumption': np.random.normal(
                comp_params['power_consumption']['mean'],
                comp_params['power_consumption']['std'],
                n_samples
            ),
            'efficiency': np.random.normal(
                comp_params['efficiency']['mean'],
                comp_params['efficiency']['std'],
                n_samples
            ),
            'compressor_leak_label': 0
        }
        
        # Adiciona padrões de falha
        failure_prob = self.params['leak_patterns']['probability']
        failure_indices = np.random.choice(n_samples, int(n_samples * failure_prob), replace=False)
        
        for idx in failure_indices:
            data['bearing_temperature'][idx] += 25
            data['oil_pressure'][idx] -= 1.8
            data['efficiency'][idx] -= 0.3
            data['compressor_leak_label'][idx] = 1
        
        df = pd.DataFrame(data)
        
        # Salva os dados se save_path for fornecido
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_path)
            print(f"Compressor data saved to {save_path}")
        
        return df