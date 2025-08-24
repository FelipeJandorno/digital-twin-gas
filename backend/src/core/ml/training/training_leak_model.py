from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load  # Importação correta das funções
import os
import pandas as pd

class LeakDetector:
    def __init__(self, model_path="models/leak_detector.joblib"):
        # Verifica se o diretório de modelos existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if os.path.exists(model_path):
            self.model = load(model_path)
        else:
            self.model = self.train_model()
            dump(self.model, model_path)  # Salva o modelo treinado

    def train_model(self):
        """Treina um modelo com dados simulados"""
        # Dados de exemplo (substitua por dados reais)
        data = pd.DataFrame({
            'pressure_drop': [0.1, 0.5, 0.8, 0.2, 0.3, 0.9],  # Queda de pressão
            'flow_rate': [100, 50, 10, 90, 80, 5],            # Vazão
            'temperature_change': [0.1, 0.3, 0.7, 0.2, 0.4, 0.8],
            'leak': [0, 1, 1, 0, 0, 1]  # Rótulo (0: normal, 1: vazamento)
        })
        
        X = data[['pressure_drop', 'flow_rate', 'temperature_change']]
        y = data['leak']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    def detect_leak(self, sensor_data):
        """
        Prediz se há vazamento com base nos dados dos sensores
        
        Args:
            sensor_data: Dicionário com:
                - pressure_drop: Queda de pressão (float)
                - flow_rate: Vazão (float)
                - temperature_change: Variação de temperatura (float)
                
        Returns:
            dict: {'leak': bool, 'probability': float, 'features': dict}
        """
        features = [
            sensor_data['pressure_drop'],
            sensor_data['flow_rate'],
            sensor_data['temperature_change']
        ]
        
        proba = self.model.predict_proba([features])[0][1]
        prediction = self.model.predict([features])[0]
        
        return {
            'leak': bool(prediction),
            'probability': float(proba),
            'features': {
                'pressure_drop': features[0],
                'flow_rate': features[1],
                'temperature_change': features[2]
            }
        }