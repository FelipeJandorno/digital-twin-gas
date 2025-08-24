from scipy import stats
import numpy as np

class DataValidator:
    def __init__(self, config_path="../data/config/generation_params.json"):
        self.config_path = config_path
    
    def validate_distribution(self, data, column, expected_mean, expected_std):
        """Valida se a distribuição está dentro dos parâmetros esperados"""
        ks_stat, p_value = stats.kstest(
            data[column], 'norm', args=(expected_mean, expected_std)
        )
        return {
            'column': column,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'within_parameters': p_value > 0.05
        }
    
    def validate_leak_ratio(self, data, expected_ratio=0.05):
        """Valida a proporção de vazamentos"""
        actual_ratio = data['leak_label'].mean()
        return {
            'expected_ratio': expected_ratio,
            'actual_ratio': actual_ratio,
            'deviation': abs(actual_ratio - expected_ratio),
            'acceptable': abs(actual_ratio - expected_ratio) < 0.01
        }
    
    def generate_validation_report(self, data, params):
        """Gera relatório completo de validação"""
        report = {
            'leak_ratio': self.validate_leak_ratio(data, params['leak_patterns']['probability'])
        }
        
        # Valida distribuições normais
        for col in ['pressure', 'flow_rate', 'temperature', 'vibration']:
            col_params = params['normal_operation'][col]
            report[col] = self.validate_distribution(
                data, col, col_params['mean'], col_params['std']
            )
        
        return report