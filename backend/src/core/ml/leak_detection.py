import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import os
from pathlib import Path

# DEFINIÇÃO FIXA das features (fora da classe para ser global)
FEATURE_COLUMNS = [
    'pressure_drop', 
    'flow_variation', 
    'temp_change',
    'hour',
    'day_of_week'
]

class LeakDetector:
    def __init__(self, model_path="models/leak_detector.joblib", 
                 data_path="../data/synthetic/generated/gas_operation_data.parquet"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model_path = model_path
        self.data_path = data_path
        self.feature_columns = FEATURE_COLUMNS
        
        if os.path.exists(model_path):
            print("Carregando modelo existente...")
            self.model = load(model_path)
            self._validate_model_features()
        else:
            print("Treinando novo modelo...")
            self.model = self._train_model()
            dump(self.model, model_path)
    
    def _load_training_data(self):
        """Carrega dados de treinamento do arquivo parquet"""
        try:
            if os.path.exists(self.data_path):
                print(f"Carregando dados de: {self.data_path}")
                data = pd.read_parquet(self.data_path)
                print(f"Dados carregados: {len(data)} amostras")
                return data
            else:
                print("Arquivo de dados não encontrado. Gerando dados sintéticos...")
                return self._generate_sample_data(10000)
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return self._generate_sample_data(5000)
    
    def _validate_model_features(self):
        """Verifica se o modelo tem as features esperadas"""
        if hasattr(self.model, 'feature_names_in_'):
            model_features = list(self.model.feature_names_in_)
            expected_features = self.feature_columns
            
            if set(model_features) != set(expected_features):
                print(f" Modelo tem features inconsistentes! Recriando...")
                print(f"   Esperado: {expected_features}")
                print(f"   Encontrado: {model_features}")
                os.remove(self.model_path)
                self.model = self._train_model()
                dump(self.model, self.model_path)
    
    def _train_model(self):
        """Treina modelo com dados do parquet ou sintéticos"""
        training_data = self._load_training_data()
        processed_data = self._preprocess_data(training_data)
        
        # Prepara features garantindo a ordem correta
        X = self._prepare_features(processed_data)
        y = processed_data['leak_label']
        
        print(f"Treinando com {len(X)} amostras")
        print(f"Features: {X.columns.tolist()}")
        print(f"Proporção de vazamentos: {y.mean():.3f}")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # Avaliação rápida
        train_score = model.score(X, y)
        print(f"Acurácia no treino: {train_score:.3f}")
        
        return model
    
    def _preprocess_data(self, df):
        """Pré-processamento básico dos dados"""
        df = df.copy()
        
        # Garante que timestamp é datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # Calcula features se as colunas base existirem
        if all(col in df.columns for col in ['pressure', 'pipeline_id']):
            df['pressure_drop'] = df.groupby('pipeline_id')['pressure'].transform(
                lambda x: x.diff().fillna(0)
            )
        
        if all(col in df.columns for col in ['flow_rate', 'pipeline_id']):
            df['flow_variation'] = df.groupby('pipeline_id')['flow_rate'].transform(
                lambda x: x.pct_change().fillna(0)
            )
        
        if all(col in df.columns for col in ['temperature', 'pipeline_id']):
            df['temp_change'] = df.groupby('pipeline_id')['temperature'].transform(
                lambda x: x.diff().fillna(0)
            )
        
        # Features temporais
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        return df
    
    def _prepare_features(self, df):
        """Garante que todas as features existem na ordem correta"""
        # Cria features faltantes com valor 0
        for feature in self.feature_columns:
            if feature not in df.columns:
                df[feature] = 0
                print(f"Feature {feature} não encontrada, preenchida com 0")
        
        # Retorna apenas as features que queremos, na ordem correta
        return df[self.feature_columns].fillna(0)
    
    def detect_leaks_batch(self, sensor_df):
        """Processa um lote de dados de sensores"""
        processed_df = self._preprocess_data(sensor_df)
        X = self._prepare_features(processed_df)
        
        # Verifica se todas as features estão presentes
        missing_features = set(self.feature_columns) - set(X.columns)
        if missing_features:
            print(f"Features faltantes: {missing_features}")
            for feature in missing_features:
                X[feature] = 0
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        results_df = processed_df.copy()
        results_df['leak_prediction'] = predictions
        results_df['leak_probability'] = probabilities
        results_df['leak_alert'] = results_df['leak_probability'] > 0.7
        
        # Estatísticas
        n_leaks = results_df['leak_alert'].sum()
        print(f"Vazamentos detectados: {n_leaks}/{len(results_df)} ({n_leaks/len(results_df):.1%})")
        
        return results_df
    
    def _generate_sample_data(self, n_samples=500):
        """Gera dados de exemplo para treinamento (fallback)"""
        print("Gerando dados sintéticos de fallback...")
        
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='min')
        
        data = {
            'timestamp': timestamps,
            'pressure': np.random.normal(80, 5, n_samples),
            'flow_rate': np.random.normal(100, 10, n_samples),
            'temperature': np.random.normal(25, 2, n_samples),
            'pipeline_id': ['pipe_1'] * n_samples,
            'location': ['station_a'] * n_samples,
            'leak_label': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        }
        
        return pd.DataFrame(data)
    
    def evaluate_model(self, test_data=None):
        """Avalia o modelo com dados de teste"""
        if test_data is None:
            test_data = self._load_training_data().sample(1000)
        
        processed_data = self._preprocess_data(test_data)
        X = self._prepare_features(processed_data)
        y = processed_data['leak_label']
        
        from sklearn.metrics import classification_report, confusion_matrix
        predictions = self.model.predict(X)
        
        print("Relatório de Classificação:")
        print(classification_report(y, predictions))
        
        return classification_report(y, predictions, output_dict=True)

# Função de utilidade para teste rápido
def test_leak_detector():
    """Testa o leak detector com dados de exemplo"""
    detector = LeakDetector()
    
    # Dados de teste
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-02', periods=100, freq='min'),
        'pressure': np.random.normal(75, 6, 100),
        'flow_rate': np.random.normal(95, 12, 100),
        'temperature': np.random.normal(24, 3, 100),
        'pipeline_id': ['test_pipe'] * 100,
        'location': ['test_station'] * 100
    })
    
    results = detector.detect_leaks_batch(test_data)
    return results

if __name__ == "__main__":
    # Teste o detector
    results = test_leak_detector()
    print(f"Teste concluído! {len(results)} amostras processadas.")