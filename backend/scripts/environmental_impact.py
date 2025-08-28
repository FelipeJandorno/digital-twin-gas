import pandas as pd
import numpy as np

# Dados fornecidos
equipamento = ["gasoduto1", "compressor1", "valvula1", "gasoduto2"]
p = [0.2, 0.5, 0.9, 0.1]

# Mapeamento JSON
json_mapping = {
    "gasodutos": 0.5,
    "compressores": 0.9,
    "valvula": 0.1
}

def process_data():
    """Processa os dados e retorna o DataFrame"""
    # Função para mapear o equipamento para a categoria do JSON
    def mapear_categoria(nome_equipamento):
        if "gasoduto" in nome_equipamento:
            return "gasodutos"
        elif "compressor" in nome_equipamento:
            return "compressores"
        elif "valvula" in nome_equipamento:
            return "valvula"
        else:
            return None

    # Criar as colunas do DataFrame
    s = [mapear_categoria(eq) for eq in equipamento]
    s_from_json = [json_mapping[categoria] for categoria in s]

    # Criar o DataFrame
    df = pd.DataFrame({
        'equipamento': equipamento,
        's': s,
        'p': p,  # p da lista fornecida
        's_json': s_from_json  #s obtido do JSON
    })
    
    return df

def calculate_impact(s, p, flow_in=100):
    """
    Calcula o fluxo de vazamento baseado nas probabilidades
    
    Parâmetros:
    s: série com categorias dos equipamentos
    p: série com probabilidades
    flow_in: fluxo de entrada (padrão = 1)
    """
    # Convertendo as categorias para valores numéricos se necessário
    # Se s for categórico, precisamos converter para valores do JSON
    if s.dtype == 'object':
        # Mapear categorias para valores do JSON
        s_numeric = s.map(json_mapping)
    else:
        s_numeric = s
    
    # Calcular o produto s * p para cada equipamento e somar
    total_impact = (s_numeric * p).sum()
    
    # Calcular o fluxo de vazamento
    flow_leaking = flow_in * (1 - total_impact)
    
    return flow_leaking

# Processar os dados
df = process_data()
print("DataFrame:")
print(df)
print()

# Calcular o impacto
result = calculate_impact(df["s"], df["p"])
print(f"Resultado do fluxo de vazamento: {result}")