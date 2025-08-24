from flask import Flask, request, jsonify, render_template
import os
import json

app = Flask(__name__)

SAVE_FOLDER = "json"
os.makedirs(SAVE_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("EcoLeak Twin.html")

@app.route('/salvar-json', methods=['POST'])
def salvar_json():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "erro", "msg": "Nenhum dado recebido"}), 400

        filepath = os.path.join(SAVE_FOLDER, "equipamentos.json")

        # sobrescreve o mesmo arquivo sempre
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return jsonify({"status": "ok", "msg": "Arquivo atualizado com sucesso!"})
    except Exception as e:
        return jsonify({"status": "erro", "msg": str(e)}), 500

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)