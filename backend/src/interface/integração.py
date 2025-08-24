// estrutura


seu-projeto/
│
├─ app.py         # Flask backend
├─ json/          # Pasta onde os arquivos JSON serão salvos
└─ templates/
   └─ index.html  # Seu front-end HTML



//nova estrutura

   
   /projeto
   /json
      equipamentos.json
   app.py
   index.html




// código básico do flask 

from flask import Flask, request, jsonify
import os
import json

app = Flask(__name__)

# pasta onde ficará o json consolidado
SAVE_FOLDER = "json"
os.makedirs(SAVE_FOLDER, exist_ok=True)

@app.route('/salvar-json', methods=['POST'])
def salvar_json():
    try:
        data = request.get_json(force=True)  # já vem agrupado do front
        if not data:
            return jsonify({"status": "erro", "msg": "Nenhum dado recebido"}), 400

        # nome fixo do arquivo
        filepath = os.path.join(SAVE_FOLDER, "equipamentos.json")

        # salva em formato bonito (indentado)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return jsonify({"status": "ok", "msg": f"Arquivo salvo em {filepath}"})
    except Exception as e:
        return jsonify({"status": "erro", "msg": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



  // código do front
  try {
    const response = await fetch('/salvar-json', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(equipamentos)
    });

    const result = await response.json();
    if(result.status === "ok"){
      alert(result.msg); // ex: "5 arquivos salvos!"
    } else {
      alert("Erro ao salvar arquivos: " + result.msg);
    }
  } catch (err) {
    alert("Erro de conexão: " + err.message);
  }
});




// correção


from flask import Flask, request, jsonify, render_template
import os, json

app = Flask(__name__)

JSON_DIR = "json"
os.makedirs(JSON_DIR, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/salvar-json', methods=['POST'])
def salvar_json():
    data = request.json
    if not data:
        return jsonify({"status": "erro", "msg": "Nenhum dado recebido"}), 400

    for eq in data:
        eq_id = eq.get("id", "sem_id")
        filename = os.path.join(JSON_DIR, f"equipamento_{eq_id}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(eq, f, ensure_ascii=False, indent=2)

    return jsonify({"status": "ok", "msg": f"{len(data)} arquivos salvos!"})

if __name__ == "__main__":
    app.run(debug=True)


