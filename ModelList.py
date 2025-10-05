import os
import json
import shutil
from datetime import datetime
from flask import Blueprint, render_template

# ------------------------------------------------------------------------------
# Blueprint Setup
# ------------------------------------------------------------------------------

models_bp = Blueprint("models", __name__, template_folder="templates")

# Path where custom models are stored
CUSTOM_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "custom_models"
)

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def ensure_custom_models_dir() -> None:
    """Ensure that the directory for storing custom models exists."""
    os.makedirs(CUSTOM_MODELS_DIR, exist_ok=True)


def list_models() -> list[str]:
    """
    Return a list of model names found in the custom_models directory.

    If there are no custom models, return ["FM"] as a default list.
    """
    ensure_custom_models_dir()
    entries = []

    for name in sorted(os.listdir(CUSTOM_MODELS_DIR)):
        path = os.path.join(CUSTOM_MODELS_DIR, name)
        # Accept directories or .json metadata files as models
        if os.path.isdir(path) or name.endswith(".json"):
            entries.append(name)

    # Normalize names: if metadata files exist, strip extension
    models = [entry[:-5] if entry.endswith(".json") else entry for entry in entries]

    return models or ["FM"]


def _safe_model_name() -> str:
    """Generate a unique, timestamp-based model name."""
    return datetime.now().strftime("FM_%Y%m%d_%H%M%S")


def create_model_entry(params: dict) -> str:
    """
    Create a new model entry under `custom_models`.

    - Writes a metadata JSON file.
    - Copies a generated ZIP file (`modelo_generado.zip`) into the model directory if it exists.
    Returns the created model name.
    """
    ensure_custom_models_dir()

    model_name = _safe_model_name()
    model_dir = os.path.join(CUSTOM_MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    metadata = {
        "name": model_name,
        "created_at": datetime.now().isoformat(),
        "params": params,
    }

    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Attempt to copy generated zip file if present
    repo_root = os.path.dirname(os.path.abspath(__file__))
    generated_zip = os.path.join(repo_root, "modelo_generado.zip")

    if os.path.exists(generated_zip):
        try:
            shutil.copy(generated_zip, os.path.join(model_dir, "model.zip"))
        except Exception as e:
            # Don't break the process if copying fails
            print(f"Warning: Failed to copy model ZIP — {e}")

    return model_name

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

@models_bp.route("/models")
def models_view():
    """Render a list of available models."""
    models = list_models()
    return render_template("models.html", models=models, datetime=datetime)


@models_bp.route("/models/create_dummy", methods=["GET"])
def create_dummy_model():
    """
    Create a dummy model entry with minimal parameters.

    Useful for testing the models UI when the real model generation flow is unavailable.
    """
    try:
        create_model_entry({"note": "Dummy created via /models/create_dummy"})
    except Exception as e:
        print(f"Error creating dummy model: {e}")

    models = list_models()
    return render_template("models.html", models=models, datetime=datetime)
