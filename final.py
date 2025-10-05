from flask import Flask, render_template, request, Response, jsonify
import pandas as pd
import subprocess
import io
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:

        file = request.files.get("file")
        if file:
            df = pd.read_csv(file)
        else:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data received"}), 400
            df = pd.DataFrame(data)

        input_path = "input_table.csv"
        output_path = "./out/predictions_outputs.csv"
        os.makedirs("./out", exist_ok=True)
        df.to_csv(input_path, index=False)

        cmd = "./run.sh"
        try:
            subprocess.run(["bash", cmd], check=True)
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"Command failed: {e}"}), 500

        if not os.path.exists(output_path):
            return jsonify({"error": "Output file not found"}), 500

        with open(output_path, "r") as f:
            output_csv = f.read()

        try:
            os.remove(input_path)
            os.remove(output_path)
        except Exception as e:
            print("⚠️ Could not remove temp files:", e)

        return Response(
            output_csv,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=results.csv"}
        )

    except Exception as e:
        print("⚠️ Server error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000)
