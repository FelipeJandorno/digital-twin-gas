import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import random

class SyntheticDataGenerator:
    def __init__(self, config_path=f"{Path(__file__).parent.parent.parent.parent}\\data\\config\\generation_params.json"):
        self.load_config(config_path)
    
    def load_config(self, config_path):
        """Carrega parâmetros de configuração"""
        with open(config_path, 'r') as f:
            self.params = json.load(f)
    
    def generate_normal_operation_data(self, n_samples=10000, start_date='2024-01-01'):
        """Gera dados de operação normal"""
        data = {
            'timestamp': pd.date_range(start_date, periods=n_samples, freq='min'),
            'pressure': np.random.normal(
                self.params['normal_operation']['pressure']['mean'],
                self.params['normal_operation']['pressure']['std'],
                n_samples
            ),
            'flow_rate': np.random.normal(
                self.params['normal_operation']['flow_rate']['mean'],
                self.params['normal_operation']['flow_rate']['std'],
                n_samples
            ),
            'temperature': np.random.normal(
                self.params['normal_operation']['temperature']['mean'],
                self.params['normal_operation']['temperature']['std'],
                n_samples
            ),
            'vibration': np.random.normal(
                self.params['normal_operation']['vibration']['mean'],
                self.params['normal_operation']['vibration']['std'],
                n_samples
            ),
            'leak_label': 0
        }
        return pd.DataFrame(data)
    
    def inject_leak_patterns(self, df):
        """Injeta padrões de vazamento nos dados"""
        df_leak = df.copy()
        n_leaks = int(len(df) * self.params['leak_patterns']['probability'])
        leak_indices = random.sample(range(len(df)), n_leaks)
        
        for idx in leak_indices:
            # Aplica padrão de vazamento
            pressure_drop = self.params['leak_patterns']['pressure_drop']
            vibration_inc = self.params['leak_patterns']['vibration_increase']
            flow_reduction = self.params['leak_patterns']['flow_reduction']
            
            df_leak.loc[idx, 'pressure'] *= (1 - pressure_drop)
            df_leak.loc[idx, 'vibration'] *= (1 + vibration_inc)
            df_leak.loc[idx, 'flow_rate'] *= (1 - flow_reduction)
            df_leak.loc[idx, 'leak_label'] = 1
            
            # Efeito cascata
            duration = self.params['leak_patterns']['effect_duration']
            for i in range(1, duration + 1):
                if idx + i < len(df):
                    decay = 1 - (i * 0.2)
                    df_leak.loc[idx+i, 'pressure'] *= (1 - pressure_drop * decay)
                    df_leak.loc[idx+i, 'vibration'] *= (1 + vibration_inc * decay)
                    df_leak.loc[idx+i, 'leak_label'] = 1 if random.random() > 0.3 else 0
        
        return df_leak
    
    def add_temporal_correlations(self, df):
        """Adiciona correlações temporais realistas"""
        df = df.copy()
        
        # Correlação pressão-vazão
        df['flow_rate'] = (df['flow_rate'] * 0.8 + 
                          df['pressure'] * 0.2 * np.random.normal(1, 0.1))
        
        # Padrões sazonais
        df['hour'] = df['timestamp'].dt.hour
        seasonal_amp = self.params['temporal_settings']['seasonal_amplitude']
        df['pressure'] = df['pressure'] * (1 + seasonal_amp * np.sin(df['hour'] * np.pi / 12))
        
        return df
    
    def generate_complete_dataset(self, n_samples=20000, save_path=None):
        """Gera dataset completo"""
        print("Generating normal operation data...")
        normal_data = self.generate_normal_operation_data(n_samples)
        
        print("Injecting leak patterns...")
        leak_data = self.inject_leak_patterns(normal_data)
        
        print("Adding temporal correlations...")
        final_data = self.add_temporal_correlations(leak_data)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            final_data.to_parquet(save_path)
            print(f"Dataset saved to {save_path}")
        
        return final_data

