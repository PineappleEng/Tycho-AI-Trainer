from flask import Flask, request, jsonify, render_template, redirect, send_file
import os
import json
import base64
from shutil import copyfile
from Predition import evaluate
import pandas as pd
import csv
# Import main from GenerateModel.py
import GenerateModel

# Import and register models blueprint
# from REPO2.ModelList import models_bp, create_model_entry
from ModelList import models_bp, create_model_entry

app = Flask(__name__)
app.register_blueprint(models_bp)

model_index = 0
# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "dataset.csv")
JSON_FILE_PATH = os.path.join(BASE_DIR, "params.json")
DEFAULT_DATASET = os.path.join(BASE_DIR, "input.csv")


# --- Disable caching (very important) ---
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "-1"
    return response


# --- Run training function ---
def run_training():
    """Ejecuta generate_model.main() y guarda los resultados."""
    try:
        print(">>> Iniciando entrenamiento del modelo...")
        model, history, metrics = GenerateModel.main()
    except Exception as e:
        print(f"❌ Error ejecutando generate_model.main(): {e}")
        return False

    try:
        # Crear directorio de reporte
        report_dir = os.path.join(BASE_DIR, 'data', 'model_report')
        os.makedirs(report_dir, exist_ok=True)

        # Guardar métricas
        metrics_path = os.path.join(report_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as mf:
            json.dump(metrics or {}, mf, indent=2, ensure_ascii=False)

        # Copiar imagen si existe
        src_img = os.path.join(BASE_DIR, 'mi_grafico.png')
        if os.path.isfile(src_img):
            dst_img = os.path.join(report_dir, 'mi_grafico.png')
            copyfile(src_img, dst_img)

        print(f"✅ Métricas guardadas en: {metrics_path}")
    except Exception as e:
        print(f"⚠️  Error guardando métricas/reportes: {e}")

    print("✅ Entrenamiento completado correctamente.")
    return True


# --- Routes ---
@app.route("/")
def index():
    return render_template("trainModel.html")


@app.route("/mrep", methods=["GET"])
def model_report_page():
    """Muestra el reporte del modelo entrenado."""
    return render_template("model_report.html", metrics=open("metrics.json"))


@app.route("/train", methods=["POST"])
def train():
    """Entrena el modelo con los parámetros recibidos y redirige al reporte."""
    data = None
    file = None

    # --- Multipart/form-data ---
    if request.content_type and "multipart/form-data" in request.content_type:
        json_data = request.form.get("json_data")
        if json_data:
            data = json.loads(json_data)
        file = request.files.get("dataset")

    # --- JSON puro ---
    elif request.is_json:
        payload = request.get_json()
        data = payload.get("params")
        csv_base64 = payload.get("csv_base64")
        if csv_base64:
            with open(CSV_FILE_PATH, "wb") as f:
                f.write(base64.b64decode(csv_base64))
    else:
        return jsonify({"error": "Formato de solicitud no soportado"}), 400

    if not data:
        return jsonify({"error": "No se recibieron los parámetros"}), 400

    # Normalizar nombres de parámetros
    if "value_size" not in data and "val_size" in data:
        data["value_size"] = data.get("val_size")

    batch_allowed = [8, 16, 32, 64]

    try:
        batch_size = int(data.get("batch_size"))
        epochs = int(data.get("epochs"))
        value_size = float(data.get("value_size"))
        test_size = float(data.get("test_size"))
    except Exception as e:
        return jsonify({"error": f"Parámetros con formato inválido: {e}"}), 400

    # Validaciones
    if batch_size not in batch_allowed:
        return jsonify({"error": "Batch size inválido"}), 400
    if not (0.1 <= value_size <= 0.5):
        return jsonify({"error": "Value size fuera de rango (0.1–0.5)."}), 400
    if not (1 <= epochs <= 100):
        return jsonify({"error": "Epoch fuera de rango (1–100)."}), 400
    if not (0.1 <= test_size <= 0.5):
        return jsonify({"error": "Test size fuera de rango (0.1–0.5)."}), 400

    # Guardar parámetros normalizados
    data["batch_size"] = batch_size
    data["epochs"] = epochs
    data["value_size"] = value_size
    data["test_size"] = test_size
    data["dataset"] = "dataset.csv"

    with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # Guardar dataset subido
    if file:
        try:
            file.save(CSV_FILE_PATH)
        except Exception as e:
            print(f"⚠️ No se pudo guardar el dataset subido: {e}")

    # --- Entrenamiento bloqueante ---
    print("Entrenando modelo... por favor espera.")
    success = run_training()
    print(f"Entrenamiento completado: {success}")

    # Registrar el modelo en la lista
    try:
        create_model_entry(data)
    except Exception as e:
        print(f"⚠️ No se pudo registrar el modelo: {e}")

    # --- Redirigir a la vista del reporte ---
    if success:
        response = redirect("/mrep?nocache=1")  # evita que el navegador use caché
        response.headers["Cache-Control"] = "no-store"
        return response
    else:
        return jsonify({"error": "Error durante el entrenamiento"}), 500


@app.route("/ModGenFin", methods=["GET"])
def ModGenFin():
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
    img_paths = ["accuracy_entrenamiento.png", "loss_entrenamiento.png"]
    return render_template("model_report.html", img_paths= img_paths, metrics=metrics)

@app.route('/image/<path:filename>')
def serve_image(filename):
    file_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return "Image not found", 404

@app.route("/prueba", methods=['POST', 'GET'])
def pruebas():
    global model_index  # para mantener el índice seleccionado

    if request.method == "POST":
        data = request.get_json()
        if not data or "index" not in data:
            return jsonify({"error": "No se recibió índice"}), 400

        model_index = int(data.get("index"))
        print(f"📦 Índice recibido: {model_index}")

        # Devolver JSON para el script del front
        return jsonify({"message": "Índice recibido", "index": model_index})

    # Si es GET, renderizar la página normal
    return render_template("index.html")


@app.route("/list", methods=['GET'])
def list():
    modelos_dir = os.path.join(BASE_DIR, "Modelos")

    # Crear carpeta si no existe
    if not os.path.exists(modelos_dir):
        os.makedirs(modelos_dir)

    # Obtener lista de archivos o subdirectorios dentro de "Modelos"
    models = [f for f in os.listdir(modelos_dir) if os.path.isdir(os.path.join(modelos_dir, f)) or f.endswith(".h5")]

    # Pasar la lista al template
    return render_template("models.html", models=models)

@app.route("/exec", methods=["POST"])
def exec():
    file = request.files.get("file")
    if (file):
        file.save("csvf.csv")
        print(f"{file.name}, {model_index+1}")
        evaluate("csvf.csv", model_index+1)
        data = open("predicciones_exoplanetas.csv","r")
    return data

# --- Run Flask ---
if __name__ == "__main__":
    app.run(debug=True, port=5001)
file_object = open("filename", "mode")