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

if __name__ == "__main__":
    app.run(debug=True)


