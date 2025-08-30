import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

FEATURE_COLUMNS_LSTM = [
    'pressure_drop', 
    'flow_variation', 
    'temp_change',
    'hour_sin', 'hour_cos',  # Features cíclicas para hora
    'day_sin', 'day_cos'     # Features cíclicas para dia da semana
]

class BaseLeakDetector:
    """Classe base com funcionalidades comuns"""
    
    def __init__(self, feature_columns, data_path):
        self.feature_columns = feature_columns
        self.data_path = data_path
    
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
    
    def _preprocess_data(self, df, for_lstm=False):
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
            if for_lstm:
                # Features cíclicas para LSTM
                df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
                df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
                df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
            else:
                # Features normais para outros modelos
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
        
        processed_data = self._preprocess_data(test_data, for_lstm=isinstance(self, LeakDetectorLSTM))
        X = self._prepare_features(processed_data)
        
        if hasattr(self, 'scaler'):
            X = self.scaler.transform(X)
        
        if hasattr(self, 'model') and hasattr(self.model, 'predict'):
            y = processed_data['leak_label']
            
            if isinstance(self, LeakDetectorLSTM):
                # Para LSTM, precisamos criar sequências
                X_seq, y_seq = self._create_sequences(X, y.values)
                predictions = (self.model.predict(X_seq, verbose=0) > 0.5).astype(int).flatten()
                probabilities = self.model.predict(X_seq, verbose=0).flatten()
                y_true = y_seq
            else:
                predictions = self.model.predict(X)
                probabilities = self.model.predict_proba(X)[:, 1]
                y_true = y
            
            print("Relatório de Classificação:")
            print(classification_report(y_true, predictions))
            print(f"AUC-ROC: {roc_auc_score(y_true, probabilities):.3f}")
            
            return classification_report(y_true, predictions, output_dict=True)
        
        return None

class LeakDetectorLogistic(BaseLeakDetector):
    def __init__(self, model_path="models/leak_detector_logistic.joblib", 
                 scaler_path="models/scaler_logistic.joblib",
                 data_path="C:\\Users\\marco\\OneDrive\\Área de Trabalho\\Hackathon\\digital-twin-gas\\backend\\data\\synthetic\\generated\\gas_operation_data.parquet"):
        
        super().__init__(FEATURE_COLUMNS, data_path)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            print("Carregando modelo e scaler existentes...")
            self.model = load(model_path)
            self.scaler = load(scaler_path)
            self._validate_model_features()
        else:
            print("Treinando novo modelo Logistic Regression...")
            self.model, self.scaler = self._train_model()
            dump(self.model, model_path)
            dump(self.scaler, scaler_path)
    
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
                os.remove(self.scaler_path)
                self.model, self.scaler = self._train_model()
                dump(self.model, self.model_path)
                dump(self.scaler, self.scaler_path)
    
    def _train_model(self):
        """Treina modelo Logistic Regression"""
        training_data = self._load_training_data()
        processed_data = self._preprocess_data(training_data)
        
        X = self._prepare_features(processed_data)
        y = processed_data['leak_label']
        
        print(f"Treinando Logistic Regression com {len(X)} amostras")
        print(f"Features: {X.columns.tolist()}")
        print(f"Proporção de vazamentos: {y.mean():.3f}")
        
        # Normalização para Logistic Regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(random_state=42, 
                                   max_iter=2000, 
                                   class_weight='balanced',
                                   solver='saga',
                                   penalty='elasticnet',
                                   l1_ratio=0.5)
        model.fit(X_scaled, y)
        
        # Avaliação rápida
        train_score = model.score(X_scaled, y)
        print(f"Acurácia no treino: {train_score:.3f}")
        
        return model, scaler
    
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
        
        # Aplica a mesma normalização
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        results_df = processed_df.copy()
        results_df['leak_prediction'] = predictions
        results_df['leak_probability'] = probabilities
        results_df['leak_alert'] = results_df['leak_probability'] > 0.7
        
        # Estatísticas
        n_leaks = results_df['leak_alert'].sum()
        print(f"Vazamentos detectados: {n_leaks}/{len(results_df)} ({n_leaks/len(results_df):.1%})")
        
        return results_df

class LeakDetectorRandomForest(BaseLeakDetector):
    def __init__(self, model_path="models/leak_detector_rf.joblib", 
                 data_path="C:\\Users\\marco\\OneDrive\\Área de Trabalho\\Hackathon\\digital-twin-gas\\backend\\data\\synthetic\\generated\\gas_operation_data.parquet"):
        
        super().__init__(FEATURE_COLUMNS, data_path)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model_path = model_path
        
        if os.path.exists(model_path):
            print("Carregando modelo Random Forest existente...")
            self.model = load(model_path)
            self._validate_model_features()
        else:
            print("Treinando novo modelo Random Forest...")
            self.model = self._train_model()
            dump(self.model, model_path)
    
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
        """Treina modelo Random Forest otimizado"""
        training_data = self._load_training_data()
        processed_data = self._preprocess_data(training_data)
        
        X = self._prepare_features(processed_data)
        y = processed_data['leak_label']
        
        print(f"Treinando Random Forest com {len(X)} amostras")
        print(f"Features: {X.columns.tolist()}")
        print(f"Proporção de vazamentos: {y.mean():.3f}")
        
        # Random Forest otimizado para detecção de anomalias
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='log2',
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X, y)
        
        # Avaliação rápida
        train_score = model.score(X, y)
        print(f"Acurácia no treino: {train_score:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Importância das features:")
        print(feature_importance)
        
        return model
    
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

class LeakDetectorLSTM(BaseLeakDetector):
    def __init__(self, model_path="models/leak_detector_lstm.h5", 
                 scaler_path="models/scaler_lstm.joblib",
                 data_path="C:\\Users\\marco\\OneDrive\\Área de Trabalho\\Hackathon\\digital-twin-gas\\backend\\data\\synthetic\\generated\\deepseek.parquet",
                 sequence_length=10):
        
        super().__init__(FEATURE_COLUMNS_LSTM, data_path)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.sequence_length = sequence_length
        
        # if os.path.exists(model_path) and os.path.exists(scaler_path):
        #     print("Carregando modelo LSTM e scaler existentes...")
        #     self.model = load_model(model_path)
        #     self.scaler = load(scaler_path)
        # else:
        print("Treinando novo modelo LSTM...")
        self.model, self.scaler = self._train_model()
        self.model.save(model_path)
        dump(self.scaler, scaler_path)
    
    def _create_sequences(self, data, labels):
        """Cria sequências para treinamento LSTM"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(labels[i + self.sequence_length - 1])
        return np.array(X), np.array(y)
    
    def _train_model(self):
        """Treina modelo LSTM com dados sequenciais"""
        training_data = self._load_training_data()
        processed_data = self._preprocess_data(training_data, for_lstm=True)
        
        # Prepara features
        X = self._prepare_features(processed_data)
        y = processed_data['leak_label'].values
        
        print(f"Treinando LSTM com {len(X)} amostras")
        print(f"Features: {X.columns.tolist()}")
        print(f"Proporção de vazamentos: {y.mean():.3f}")
        
        # Normalização
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cria sequências
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
        )
        
        print(f"Sequências de treino: {X_train.shape}")
        print(f"Sequências de teste: {X_test.shape}")
        
        # Modelo LSTM
        model = Sequential([
            Input(shape=(self.sequence_length, len(self.feature_columns))),
            LSTM(64, return_sequences=True, dropout=0.2),
            LSTM(32, dropout=0.2),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(self.model_path, save_best_only=True)
        ]
        
        # Treinamento
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Avaliação
        train_loss, train_acc, train_prec, train_rec = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Acurácia no treino: {train_acc:.3f}, Teste: {test_acc:.3f}")
        print(f"Precisão no treino: {train_prec:.3f}, Teste: {test_prec:.3f}")
        print(f"Recall no treino: {train_rec:.3f}, Teste: {test_rec:.3f}")
        
        return model, scaler
    
    def detect_leaks_batch(self, sensor_df):
        """Processa um lote de dados de sensores para LSTM"""
        processed_df = self._preprocess_data(sensor_df, for_lstm=True)
        X = self._prepare_features(processed_df)
        
        # Verifica se todas as features estão presentes
        missing_features = set(self.feature_columns) - set(X.columns)
        if missing_features:
            print(f"Features faltantes: {missing_features}")
            for feature in missing_features:
                X[feature] = 0
        
        # Normalização
        X_scaled = self.scaler.transform(X)
        
        # Cria sequências para predição
        if len(X_scaled) >= self.sequence_length:
            X_seq = np.array([X_scaled[i:i + self.sequence_length] 
                            for i in range(len(X_scaled) - self.sequence_length + 1)])
            
            # Predições
            probabilities = self.model.predict(X_seq, verbose=0).flatten()
            predictions = (probabilities > 0.5).astype(int)
            
            # Preenche as primeiras sequências com NaN
            pad_length = self.sequence_length - 1
            pad_probs = np.full(pad_length, np.nan)
            pad_preds = np.full(pad_length, -1)
            
            probabilities = np.concatenate([pad_probs, probabilities])
            predictions = np.concatenate([pad_preds, predictions])
        else:
            print("Dados insuficientes para criar sequências LSTM")
            probabilities = np.full(len(processed_df), np.nan)
            predictions = np.full(len(processed_df), -1)
        
        results_df = processed_df.copy()
        results_df['leak_prediction'] = predictions
        results_df['leak_probability'] = probabilities
        results_df['leak_alert'] = results_df['leak_probability'] > 0.7
        
        # Estatísticas (ignorando valores NaN)
        valid_leaks = results_df['leak_alert'].dropna().sum()
        valid_total = results_df['leak_alert'].dropna().count()
        
        if valid_total > 0:
            print(f"Vazamentos detectados: {valid_leaks}/{valid_total} ({valid_leaks/valid_total:.1%})")
        
        return results_df

# Funções de utilidade para teste
def test_all_models():
    """Testa todos os modelos com dados de exemplo"""
    
    # Dados de teste
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-02', periods=200, freq='min'),
        'pressure': np.random.normal(75, 6, 200),
        'flow_rate': np.random.normal(95, 12, 200),
        'temperature': np.random.normal(24, 3, 200),
        'pipeline_id': ['test_pipe'] * 200,
        'location': ['test_station'] * 200
    })
    
    # Testar Logistic Regression
    print("=" * 50)
    print("TESTANDO LOGISTIC REGRESSION")
    print("=" * 50)
    detector_lr = LeakDetectorLogistic()
    results_lr = detector_lr.detect_leaks_batch(test_data)
    
    # Testar Random Forest
    print("\n" + "=" * 50)
    print("TESTANDO RANDOM FOREST")
    print("=" * 50)
    detector_rf = LeakDetectorRandomForest()
    results_rf = detector_rf.detect_leaks_batch(test_data)
    
    # Testar LSTM (com mais dados)
    print("\n" + "=" * 50)
    print("TESTANDO LSTM")
    print("=" * 50)
    detector_lstm = LeakDetectorLSTM(sequence_length=5)
    results_lstm = detector_lstm.detect_leaks_batch(test_data)
    
    return {
        'logistic_regression': results_lr,
        'random_forest': results_rf,
        'lstm': results_lstm
    }

def compare_models():
    """Compara o desempenho de todos os modelos"""
    print("COMPARANDO MODELOS")
    print("=" * 60)
    
    # Carregar dados de teste
    data_path = "C:\\Users\\marco\\OneDrive\\Área de Trabalho\\Hackathon\\digital-twin-gas\\backend\\data\\synthetic\\generated\\gas_operation_data.parquet"
    
    try:
        if os.path.exists(data_path):
            test_data = pd.read_parquet(data_path).sample(1000)
        else:
            # Gerar dados sintéticos se não existir
            base_detector = BaseLeakDetector(FEATURE_COLUMNS, data_path)
            test_data = base_detector._generate_sample_data(1000)
    except:
        base_detector = BaseLeakDetector(FEATURE_COLUMNS, data_path)
        test_data = base_detector._generate_sample_data(1000)
    
    # Avaliar cada modelo
    models = {
        'Logistic Regression': LeakDetectorLogistic(),
        'Random Forest': LeakDetectorRandomForest()
        # 'LSTM': LeakDetectorLSTM(sequence_length=5)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 30)
        try:
            report = model.evaluate_model(test_data)
            results[name] = report
        except Exception as e:
            print(f"Erro ao avaliar {name}: {e}")
            results[name] = None
    
    return results

if __name__ == "__main__":
    # Testar todos os modelos
    results = test_all_models()
    print(f"\nTeste concluído! Resultados disponíveis para {len(results)} modelos.")
    
    # Comparar desempenho
    comparison = compare_models()
    print("\nComparação finalizada!")