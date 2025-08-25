from flask import Flask, request, jsonify, render_template
import os, json

app = Flask(__name__)

JSON_DIR = "json"
os.makedirs(JSON_DIR, exist_ok=True)

@app.route("/")
def home():
    return render_template("EcoLeak Twin.html")

@app.route('/salvar-json', methods=['POST'])
def salvar_json():
    data = request.get_json(force=True)

    if not isinstance(data, dict):
        return jsonify({"status": "erro", "msg": "Formato inválido"}), 400

    filepath = os.path.join(JSON_DIR, "equipamentos.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


    return jsonify({"status": "ok", "msg": f"{len(data)} arquivos salvos!"})

@app.route('/enviar-json', methods=["POST", "GET"])
def enviar_json():
    try:
        # Carrega o arquivo JSON salvo
        filepath = os.path.join(JSON_DIR, "resposta_usuario.json")
        
        if not os.path.exists(filepath):
            return jsonify({"status": "erro", "msg": "Arquivo não encontrado"}), 404
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return jsonify({
            "status": "ok", 
            "data": data,
            "msg": "Dados enviados com sucesso"
        })
        
    except Exception as e:
        return jsonify({"status": "erro", "msg": f"Erro ao ler arquivo: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)


# Abaixo encontra-se a função .js a ser adicionada no frontend para receber os dados do backend

# // Recebe os dados do backend
# async function recebeDados(){
#     try {
#         const response = await fetch('http://127.0.0.1:5000/enviar-json', {
#             method: 'GET',
#             headers: { 
#                 'Accept': 'application/json'
#             }
#         });
        
#         // Verifique o content-type antes de fazer .json()
#         const contentType = response.headers.get('content-type');
        
#         if (!contentType || !contentType.includes('application/json')) {
#             const text = await response.text();
#             console.error('Resposta não é JSON:', text.substring(0, 200));
#             alert('Servidor retornou HTML em vez de JSON. Verifique a rota.');
#             return;
#         }
        
#         const result = await response.json();

#         if(result.status === "ok"){
#             alert(JSON.stringify(result.data));
#         } else {
#             alert("Erro: " + result.msg);
#         }
#     } catch (err) {
#         alert("Erro de conexão: " + err.message);
#     }
# }