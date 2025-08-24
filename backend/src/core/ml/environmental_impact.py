import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump, load
import os

class ImpactCalculator:
    def __init__(self, model_path="models/impact_calculator.joblib"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if os.path.exists(model_path):
            self.model = load(model_path)
        else:
            self.model = self._train_model()  # Método correto
            dump(self.model, model_path)
    
    def _train_model(self):  # Nome correto do método
        # Dados simulados (emissões em toneladas de CO2eq)
        X = np.array([[10], [50], [100]])  # Volume de gás vazado (m³)
        y = np.array([0.5, 2.5, 5.0])      # Emissões equivalentes
        
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    def calculate_impact(self, gas_volume):
        return float(self.model.predict([[gas_volume]])[0])