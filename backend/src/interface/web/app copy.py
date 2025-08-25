from flask import Flask, request, jsonify, send_file, render_template
import json
import os
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

SAVE_FOLDER = "json"
os.makedirs(SAVE_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("EcoLeak Twin.html")

app = Flask(__name__)
CORS(app)

# Dados fixos que serão retornados
DADOS_FIXOS = {
    "gasodutos": {
        "Id": "id_equipamento",
        "% vazamento": "10%",
        "Metano": "100 und",
        "CO2 eq": "150 und"
    },
    "compressores": {
        "% vazamento": "5%", 
        "CO2 eq": "200 und"
    },
    "timestamp": "default",
    "total_equipamentos": "23"
}

@app.route('/salvar-json', methods=['POST'])
def salvar_json():
    try:
        # Recebe dados do frontend
        dados = request.get_json()
        
        print("Dados recebidos do frontend:", dados)
        
        # Cria cópia dos dados fixos
        dados_retorno = DADOS_FIXOS.copy()
        
        # Personaliza com dados recebidos
        if dados and 'id_equipamento' in dados:
            dados_retorno['gasodutos']['Id'] = dados['id_equipamento']
        
        # Atualiza timestamp
        dados_retorno['timestamp'] = datetime.now().isoformat()
        
        # Salva o arquivo JSON
        caminho_arquivo = 'resposta_usuario.json'
        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
            json.dump(dados_retorno, f, ensure_ascii=False, indent=4)
        
        print("Retornando dados para o frontend:", dados_retorno)
        
        return jsonify({
            'status': 'ok',
            'msg': 'Análise concluída com sucesso!',
            'dados': dados_retorno
        }), 200
        
    except Exception as e:
        print("Erro:", str(e))
        return jsonify({
            'status': 'erro',
            'msg': f'Erro ao processar análise: {str(e)}'
        }), 500

@app.route('/download-json')
def download_json():
    try:
        return send_file('resposta_usuario.json', 
                       as_attachment=True, 
                       download_name='analise_equipamentos.json',
                       mimetype='application/json')
    except Exception as e:
        return jsonify({'erro': str(e)}), 404

@app.route('/')
def index():
    return send_file('TESTE.html')  # Se quiser servir o HTML pelo Flask

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)