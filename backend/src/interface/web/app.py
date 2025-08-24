from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Permite requisições de origens diferentes

# Armazenamento em memória (em produção, use um banco de dados)
equipamentos = []

@app.route('/api/equipamentos', methods=['GET', 'POST'])
def gerenciar_equipamentos():
    if request.method == 'GET':
        # Retorna a lista de equipamentos
        return jsonify(equipamentos)
    
    elif request.method == 'POST':
        try:
            # Adiciona um novo equipamento
            novo_equipamento = request.get_json()
            novo_equipamento['id'] = f"E{len(equipamentos) + 1}"
            novo_equipamento['data_criacao'] = datetime.now().isoformat()
            equipamentos.append(novo_equipamento)
            
            return jsonify({
                "status": "success",
                "message": "Equipamento adicionado com sucesso",
                "equipamento": novo_equipamento
            }), 201
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Erro ao processar equipamento: {str(e)}"
            }), 400

@app.route('/api/equipamentos/<equipamento_id>', methods=['DELETE'])
def remover_equipamento(equipamento_id):
    global equipamentos
    equipamentos = [e for e in equipamentos if e['id'] != equipamento_id]
    
    return jsonify({
        "status": "success",
        "message": f"Equipamento {equipamento_id} removido com sucesso"
    })

@app.route('/api/impacto-ambiental', methods=['POST'])
def calcular_impacto_ambiental():
    try:
        dados = request.get_json()
        # Aqui você implementaria a lógica real de cálculo de impacto ambiental
        # Esta é uma implementação simplificada para demonstração
        
        fatores_tipo = {
            "Separador": 1.2, 
            "Filtro": 0.8, 
            "Compressor": 2.0, 
            "Válvula": 0.5, 
            "Tubulação": 0.6
        }
        
        impacto_total = 0
        co2_maximo = 0
        ch4_maximo = 0
        
        for equipamento in dados.get('equipamentos', []):
            # Cálculo baseado no tipo de equipamento
            impacto_total += fatores_tipo.get(equipamento['tipo'], 1.0)
            
            # Análise da composição do gás
            if 'gas' in equipamento:
                composicoes = equipamento['gas'].split(',')
                for composicao in composicoes:
                    if '=' in composicao:
                        componente, valor = composicao.split('=')
                        componente = componente.strip().upper()
                        valor = float(valor.strip())
                        
                        if componente == 'CO2':
                            co2_maximo = max(co2_maximo, valor)
                        elif componente == 'CH4':
                            ch4_maximo = max(ch4_maximo, valor)
        
        # Ajuste do impacto baseado na composição do gás
        impacto_ajustado = impacto_total * (1 + co2_maximo/100)
        
        return jsonify({
            "status": "success",
            "impacto_total": round(impacto_ajustado, 2),
            "co2_maximo": co2_maximo,
            "ch4_maximo": ch4_maximo,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Erro ao calcular impacto ambiental: {str(e)}"
        }), 400

@app.route('/api/exportar-relatorio', methods=['POST'])
def exportar_relatorio():
    try:
        dados = request.get_json()
        
        # Em uma implementação real, você geraria um PDF aqui
        # Esta é uma versão simplificada que retorna JSON
        relatorio = {
            "titulo": "Relatório de Equipamentos - Estação de Compressão",
            "data_geracao": datetime.now().isoformat(),
            "total_equipamentos": len(dados.get('equipamentos', [])),
            "equipamentos": dados.get('equipamentos', []),
            "impacto_ambiental": dados.get('impacto_ambiental', {})
        }
        
        # Salvar em um arquivo (simulação)
        with open('relatorio_temp.json', 'w') as f:
            json.dump(relatorio, f, indent=2)
        
        return jsonify({
            "status": "success",
            "message": "Relatório gerado com sucesso",
            "relatorio": relatorio
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Erro ao gerar relatório: {str(e)}"
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)