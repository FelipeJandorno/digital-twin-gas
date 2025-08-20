import json
from core.models.gasoduto import Gasoduto

def carregar_config(config_path="data/config/plant_default.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def salvar_config(config, config_path):
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

def adicionar_gasoduto(config):
    origem = input("Nó de origem: ")
    destino = input("Nó de destino: ")
    diametro = float(input("Diâmetro (polegadas): "))
    comprimento = float(input("Comprimento (km): "))
    pressao = float(input("Pressão (bar): "))
    
    novo_gasoduto = {
        "origem": origem, "destino": destino,
        "diametro": diametro, "comprimento": comprimento, "pressao": pressao
    }
    config["gasodutos"].append(novo_gasoduto)
    return config

if __name__ == "__main__":
    print("Editor de Topologia da Rede de Gás")
    config = carregar_config()
    print(f"Configuração atual: {config}")