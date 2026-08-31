import os
import json
import uuid
import datetime
from io import BytesIO

import numpy as np
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///local.db")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

db = SQLAlchemy(app)


class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    probabilities = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "prediction": self.prediction,
            "confidence": round(self.confidence * 100, 2),
            "probabilities": {k: round(v * 100, 2) for k, v in self.probabilities.items()},
            "created_at": self.created_at.isoformat(),
        }


with app.app_context():
    db.create_all()

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
session = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"))

with open(os.path.join(MODEL_DIR, "labels.json")) as f:
    LABELS = json.load(f)

with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

IMG_SIZE = METRICS.get("img_size", 224)


def preprocess(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]


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

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
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

    prob_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    record = Prediction(
        filename=file.filename,
        prediction=pred_label,
        confidence=confidence,
        probabilities=prob_dict,
    )
    db.session.add(record)
    db.session.commit()

    return jsonify(
        {
            "prediction": pred_label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {k: round(v * 100, 2) for k, v in prob_dict.items()},
            "id": record.id,
        }
    )


@app.route("/api/history")
def history():
    records = Prediction.query.order_by(Prediction.created_at.desc()).limit(50).all()
    return jsonify([r.to_dict() for r in records])


@app.route("/api/stats")
def stats():
    total = Prediction.query.count()
    from sqlalchemy import func

    class_counts = (
        db.session.query(Prediction.prediction, func.count(Prediction.prediction))
        .group_by(Prediction.prediction)
        .all()
    )
    return jsonify(
        {
            "total_predictions": total,
            "class_distribution": {label: count for label, count in class_counts},
            "model_accuracy": METRICS.get("test_accuracy"),
            "classes": LABELS,
        }
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
