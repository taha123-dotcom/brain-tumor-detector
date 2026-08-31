import os
import json
from io import BytesIO

import numpy as np
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

IMG_SIZE = 224
HEALTHY_THRESHOLD = 0.30
LABELS = []
METRICS = {}
session = None


def load_model():
    global session, LABELS, METRICS, IMG_SIZE, HEALTHY_THRESHOLD

    model_path = os.path.join(MODEL_DIR, "model.onnx")
    if not os.path.exists(model_path):
        for alt in [
            os.path.join(os.getcwd(), "models", "model.onnx"),
            os.path.join(os.getcwd(), "brain_tumor_model.onnx"),
        ]:
            if os.path.exists(alt):
                model_path = alt
                break

    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path)

    with open(os.path.join(MODEL_DIR, "labels.json")) as f:
        LABELS = json.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        METRICS = json.load(f)

    IMG_SIZE = METRICS.get("img_size", 224)
    HEALTHY_THRESHOLD = METRICS.get("healthy_fallback", {}).get("threshold", 0.30)
    print(f"Model ready. Classes: {LABELS}")


def preprocess(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]


load_model()


@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    image_bytes = file.read()
    input_tensor = preprocess(image_bytes)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]

    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]
    confidence = float(probs[pred_idx])

    if confidence < HEALTHY_THRESHOLD:
        pred_label = "healthy"

    prob_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    prob_dict["healthy"] = max(0, 1.0 - float(np.max(probs)))

    return jsonify({
        "prediction": pred_label,
        "confidence": round(confidence * 100, 2),
        "probabilities": {k: round(v * 100, 2) for k, v in prob_dict.items()},
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
