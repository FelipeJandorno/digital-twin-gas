#!/usr/bin/env python3
"""
Script para geração de dados sintéticos
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date


# Debug: print current directory and check if src exists
current_dir = Path(__file__).parent
print(f"Current directory: {current_dir}")

# Check both possibilities
src_path_option1 = current_dir / 'src'
src_path_option2 = current_dir.parent / 'src'

print(f"Option 1 path: {src_path_option1}")
print(f"Option 1 exists: {src_path_option1.exists()}")
print(f"Option 2 path: {src_path_option2}")
print(f"Option 2 exists: {src_path_option2.exists()}")

# Use the correct one
if src_path_option1.exists():
    src_path = src_path_option1
elif src_path_option2.exists():
    src_path = src_path_option2
else:
    raise FileNotFoundError("Could not find src directory")

if str(src_path) not in sys.path:
    sys.path.append(str(src_path))
    print(f"Added to path: {src_path}")

# Now try your imports
try:
    from core.data_generation.synthetic_data_generator import SyntheticDataGenerator
    from core.data_generation.compressor_data import CompressorDataGenerator
    from core.data_generation.validation import DataValidator
    print("Imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

print("Current Directory {} ".format(current_dir))

def main():
    print("Iniciando geração de dados sintéticos...")
    
    # Definir caminhos corretamente
    backend_dir = current_dir.parent
    data_dir = backend_dir / "data"
    
    print(f"Backend directory: {backend_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Data directory exists: {data_dir.exists()}")
    
    # Verificar conteúdo da pasta data
    if data_dir.exists():
        print("Conteúdo da pasta data:")
        for item in data_dir.iterdir():
            print(f"   → {item.name}")
    
    # Verificar se a pasta config existe
    config_dir = data_dir / "config"
    print(f"Config directory: {config_dir}")
    print(f"Config directory exists: {config_dir.exists()}")
    
    if config_dir.exists():
        print("Conteúdo da pasta config:")
        for item in config_dir.iterdir():
            print(f"   → {item.name}")
    
    # Caminho correto para o arquivo de configuração
    config_path = config_dir / "generation_params.json"
    print(f"Config file path: {config_path}")
    print(f"Config file exists: {config_path.exists()}")
    
    # Criar arquivo de configuração se não existir
    if not config_path.exists():
        print(f"Arquivo de configuração não encontrado. Criando: {config_path}")
        
        # Garantir que a pasta config existe
        config_dir.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            "validation_thresholds": {
                "pressure_min": 0.5,
                "pressure_max": 15.0,
                "flow_min": 0.1,
                "flow_max": 100.0,
                "temperature_min": -10.0,
                "temperature_max": 60.0
            },
            "data_quality": {
                "max_null_percentage": 0.01,
                "value_ranges": {
                    "pressure": [0.5, 15.0],
                    "flow_rate": [0.1, 100.0],
                    "temperature": [-10.0, 60.0]
                }
            },
            "leak_parameters": {
                "leak_probability": 0.05,
                "leak_intensity_range": [0.1, 0.8]
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        print("Arquivo de configuração criado com sucesso!")
    else:
        print("Arquivo de configuração encontrado!")
    
    # Ler configuração
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        print("Configuração carregada com sucesso!")
        print(f"Parâmetros: {json.dumps(params, indent=2)}")
    except Exception as e:
        print(f"Erro ao ler arquivo de configuração: {e}")
        return
    
    # Criar outras pastas necessárias
    (data_dir / "synthetic" / "generated").mkdir(parents=True, exist_ok=True)
    (data_dir / "synthetic" / "validated").mkdir(parents=True, exist_ok=True)
    
    # Gera dados principais
    print("Gerando dados principais...")
    try:
        generator = SyntheticDataGenerator()
        main_data_path = data_dir / "synthetic" / "generated" / "gas_operation_data.parquet"
        
        main_data = generator.generate_complete_dataset(
            n_samples=50000,
            save_path=str(main_data_path)
        )
        print("Dados principais gerados com sucesso!")
    except Exception as e:
        print(f"Erro ao gerar dados principais: {e}")
        return
    
    # Gera dados de compressores
    # print("Gerando dados de compressores...")
    # try:
    #     comp_generator = CompressorDataGenerator()
    #     comp_data_path = data_dir / "synthetic" / "generated" / "compressor_data.parquet"
        
    #     comp_data = comp_generator.generate_compressor_data(
    #         n_samples=10000,
    #         save_path=str(comp_data_path)
    #     )
    #     print("Dados de compressores gerados com sucesso!")
    # except Exception as e:
    #     print(f"Erro ao gerar dados de compressores: {e}")
    #     return
    
    # Valida dados
    print("Validando dados...")
    try:
        validator = DataValidator()
        validation_report = validator.generate_validation_report(main_data, params)
        print("Validação concluída com sucesso!")
    except Exception as e:
        print(f"Erro na validação: {e}")
        return
    
    # Salva relatório
    report_path = data_dir / "synthetic" / "validated" / "validation_report.json"
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            # Solução mais simples - converte automaticamente tipos não serializáveis
            json.dump(validation_report, f, indent=2, ensure_ascii=False, default=str)
        print(f"Relatório salvo em: {report_path}")
    except Exception as e:
        print(f"Erro ao salvar relatório: {e}")
        return
    
    print("Geração de dados concluída com sucesso!")
    
    # Estatísticas finais
    if hasattr(main_data, 'columns') and 'leak_label' in main_data.columns:
        leak_percentage = main_data['leak_label'].mean() * 100
        print(f"Proporção de vazamentos: {leak_percentage:.2f}%")
    
    print(f"Dados salvos em: {data_dir / 'synthetic' / 'generated'}")
    print(f"Relatório salvo em: {report_path}")

if __name__ == "__main__":
    main()