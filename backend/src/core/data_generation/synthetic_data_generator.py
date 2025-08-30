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
            'leak_label': 0,  # Todas as linhas inicialmente sem vazamento
            'leak_status': 'sem vazamento'  # Classificação textual
        }
        return pd.DataFrame(data)
    
    def inject_leak_patterns(self, df):
        """Injeta padrões de vazamento nos dados e atualiza a classificação"""
        df_leak = df.copy()
        n_leaks = int(len(df) * self.params['leak_patterns']['probability'])
        leak_indices = random.sample(range(len(df)), n_leaks)
        
        print(f"Injetando {n_leaks} padrões de vazamento...")
        
        for idx in leak_indices:
            # Aplica padrão de vazamento
            pressure_drop = self.params['leak_patterns']['pressure_drop']
            vibration_inc = self.params['leak_patterns']['vibration_increase']
            flow_reduction = self.params['leak_patterns']['flow_reduction']
            
            # Modifica os valores dos sensores
            df_leak.loc[idx, 'pressure'] *= (1 - pressure_drop)
            df_leak.loc[idx, 'vibration'] *= (1 + vibration_inc)
            df_leak.loc[idx, 'flow_rate'] *= (1 - flow_reduction)
            
            # Atualiza a classificação para "com vazamento"
            df_leak.loc[idx, 'leak_label'] = 1
            df_leak.loc[idx, 'leak_status'] = 'com vazamento'
            
            # Efeito cascata - pontos próximos também são afetados
            duration = self.params['leak_patterns']['effect_duration']
            for i in range(1, duration + 1):
                if idx + i < len(df):
                    decay = 1 - (i * 0.2)  # Efeito decai com o tempo
                    
                    # Aplica efeito reduzido nos sensores
                    df_leak.loc[idx+i, 'pressure'] *= (1 - pressure_drop * decay)
                    df_leak.loc[idx+i, 'vibration'] *= (1 + vibration_inc * decay)
                    
                    # Classifica como vazamento com probabilidade decrescente
                    if random.random() > 0.3:  # 70% de chance de ser classificado como vazamento
                        df_leak.loc[idx+i, 'leak_label'] = 1
                        df_leak.loc[idx+i, 'leak_status'] = 'com vazamento'
        
        return df_leak
    
    def add_temporal_correlations(self, df):
        """Adiciona correlações temporais realistas"""
        df = df.copy()
        
        # Correlação pressão-vazão
        df['flow_rate'] = (df['flow_rate'] * 0.8 + 
                          df['pressure'] * 0.2 * np.random.normal(1, 0.1))
        
        # Padrões sazonais
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        seasonal_amp = self.params['temporal_settings']['seasonal_amplitude']
        df['pressure'] = df['pressure'] * (1 + seasonal_amp * np.sin(df['hour'] * np.pi / 12))
        
        return df
    
    def add_engineered_features(self, df):
        """Adiciona features de engenharia para melhor detecção"""
        df = df.copy()
        
        # Features de taxa de variação
        df['pressure_drop'] = df['pressure'].diff().fillna(0)
        df['flow_variation'] = df['flow_rate'].pct_change().fillna(0)
        df['temp_change'] = df['temperature'].diff().fillna(0)
        df['vibration_change'] = df['vibration'].diff().fillna(0)
        
        # Médias móveis para detectar tendências
        df['pressure_ma_5'] = df['pressure'].rolling(window=5, min_periods=1).mean()
        df['flow_ma_5'] = df['flow_rate'].rolling(window=5, min_periods=1).mean()
        
        # Razões entre sensores
        df['pressure_flow_ratio'] = df['pressure'] / (df['flow_rate'] + 1e-6)  # Evita divisão por zero
        df['vibration_pressure_ratio'] = df['vibration'] / (df['pressure'] + 1e-6)
        
        return df
    
    def generate_statistics(self, df):
        """Gera estatísticas sobre o dataset"""
        total_samples = len(df)
        leak_samples = df['leak_label'].sum()
        no_leak_samples = total_samples - leak_samples
        
        print(f"\n=== ESTATÍSTICAS DO DATASET ===")
        print(f"Total de amostras: {total_samples}")
        print(f"Com vazamento: {leak_samples} ({leak_samples/total_samples:.2%})")
        print(f"Sem vazamento: {no_leak_samples} ({no_leak_samples/total_samples:.2%})")
        
        # Estatísticas por sensor para vazamentos vs não vazamentos
        if leak_samples > 0:
            print(f"\nMédias por status:")
            for sensor in ['pressure', 'flow_rate', 'temperature', 'vibration']:
                leak_mean = df[df['leak_label'] == 1][sensor].mean()
                no_leak_mean = df[df['leak_label'] == 0][sensor].mean()
                print(f"{sensor}: {no_leak_mean:.2f} (sem) vs {leak_mean:.2f} (com)")
    
    def generate_complete_dataset(self, n_samples=20000, save_path=None):
        """Gera dataset completo com classificação de vazamento"""
        print("Generating normal operation data...")
        normal_data = self.generate_normal_operation_data(n_samples)
        
        print("Injecting leak patterns...")
        leak_data = self.inject_leak_patterns(normal_data)
        
        print("Adding temporal correlations...")
        final_data = self.add_temporal_correlations(leak_data)
        
        print("Adding engineered features...")
        final_data = self.add_engineered_features(final_data)
        
        # Gera estatísticas
        self.generate_statistics(final_data)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Salva em múltiplos formatos
            final_data.to_parquet(save_path)
            final_data.to_csv(save_path.replace('.parquet', '.csv'), index=False)
            
            print(f"\nDataset salvo em:")
            print(f"- Parquet: {save_path}")
            print(f"- CSV: {save_path.replace('.parquet', '.csv')}")
        
        return final_data

# Exemplo de uso
if __name__ == "__main__":
    # Configuração padrão do gerador
    generator = SyntheticDataGenerator()
    
    # Gera dataset completo
    dataset = generator.generate_complete_dataset(
        n_samples=50000,
        save_path="C:\\Users\\marco\\OneDrive\\Área de Trabalho\\Hackathon\\digital-twin-gas\\backend\\data\\synthetic\\generated\\gas_operation_data.parquet"
    )
    
    # Exemplo de como visualizar os dados
    print("\nPrimeiras 10 linhas do dataset:")
    print(dataset[['timestamp', 'pressure', 'flow_rate', 'leak_label', 'leak_status']].head(10))
    
    print("\nAmostras com vazamento:")
    leaks = dataset[dataset['leak_label'] == 1]
    print(leaks[['timestamp', 'pressure', 'flow_rate', 'leak_status']].head(5))