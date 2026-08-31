import os
import json
import uuid
import datetime
import traceback
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

DB_URL = os.environ.get("DATABASE_URL", "")

db_engine = None
db_conn = None


def get_db_engine():
    global db_engine, db_conn
    if not DB_URL:
        print("No DATABASE_URL set, DB disabled")
        return None
    try:
        from sqlalchemy import create_engine, text, Column, String, Float, DateTime, JSON
        from sqlalchemy.orm import declarative_base, sessionmaker

        url = DB_URL.replace("postgres://", "postgresql://", 1)
        for bad in ["channel_binding=require", "&channel_binding=require", "?channel_binding=require"]:
            url = url.replace(bad, "")
        url = url.rstrip("?&")

        engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={"sslmode": "require"},
        )
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            val = result.scalar()
            print(f"DB test query: {val}")

        db_engine = engine
        return engine
    except Exception as e:
        print(f"DB connection failed: {e}")
        traceback.print_exc()
        return None


def init_db_table(engine):
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id VARCHAR(36) PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                prediction VARCHAR(50) NOT NULL,
                confidence FLOAT NOT NULL,
                probabilities JSON NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()
    print("DB table ready")


def save_prediction(engine, filename, prediction, confidence, probabilities):
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO predictions (id, filename, prediction, confidence, probabilities, created_at)
                VALUES (:id, :filename, :prediction, :confidence, :probabilities, :created_at)
            """), {
                "id": str(uuid.uuid4()),
                "filename": filename,
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": json.dumps(probabilities),
                "created_at": datetime.datetime.utcnow(),
            })
            conn.commit()
        return True
    except Exception as e:
        print(f"DB write failed: {e}")
        traceback.print_exc()
        return False


def get_history(engine, limit=50):
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT id, filename, prediction, confidence, probabilities, created_at "
                "FROM predictions ORDER BY created_at DESC LIMIT :lim"
            ), {"lim": limit})
            rows = result.fetchall()
            records = []
            for row in rows:
                probs = row[4]
                if isinstance(probs, str):
                    probs = json.loads(probs)
                records.append({
                    "id": row[0],
                    "filename": row[1],
                    "prediction": row[2],
                    "confidence": round(float(row[3]) * 100, 2),
                    "probabilities": {k: round(v * 100, 2) for k, v in probs.items()} if isinstance(probs, dict) else probs,
                    "created_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                })
            return records
    except Exception as e:
        print(f"History query failed: {e}")
        traceback.print_exc()
        return []


def get_stats(engine):
    from sqlalchemy import text
    total = 0
    class_counts = {}
    try:
        with engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar() or 0
            rows = conn.execute(text(
                "SELECT prediction, COUNT(*) as cnt FROM predictions GROUP BY prediction"
            )).fetchall()
            class_counts = {row[0]: row[1] for row in rows}
    except Exception as e:
        print(f"Stats query failed: {e}")
    return {"total_predictions": total, "class_distribution": class_counts}


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
    print(f"Model loaded. Inputs: {session.get_inputs()[0].shape}")

    with open(os.path.join(MODEL_DIR, "labels.json")) as f:
        LABELS = json.load(f)

    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        METRICS = json.load(f)

    IMG_SIZE = METRICS.get("img_size", 224)
    HEALTHY_THRESHOLD = METRICS.get("healthy_fallback", {}).get("threshold", 0.30)
    print(f"Labels: {LABELS}, Threshold: {HEALTHY_THRESHOLD}")


def preprocess(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]


load_model()
db_engine = get_db_engine()
if db_engine:
    init_db_table(db_engine)


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
    prob_dict["healthy"] = round(max(0, 1.0 - float(np.max(probs))) * 100, 2)

    saved = False
    if db_engine:
        saved = save_prediction(db_engine, file.filename, pred_label, confidence, prob_dict)

    return jsonify({
        "prediction": pred_label,
        "confidence": round(confidence * 100, 2),
        "probabilities": {k: round(v * 100, 2) if isinstance(v, float) else v for k, v in prob_dict.items()},
        "db_saved": saved,
    })


@app.route("/api/history")
def history():
    if not db_engine:
        return jsonify([])
    return jsonify(get_history(db_engine))


@app.route("/api/stats")
def stats():
    total = 0
    class_counts = {}
    if db_engine:
        s = get_stats(db_engine)
        total = s["total_predictions"]
        class_counts = s["class_distribution"]
    return jsonify({
        "total_predictions": total,
        "class_distribution": class_counts,
        "model_accuracy": METRICS.get("test_accuracy"),
        "classes": LABELS,
    })


@app.route("/api/debug")
def debug():
    return jsonify({
        "db_connected": db_engine is not None,
        "database_url_set": bool(DB_URL),
        "model_loaded": session is not None,
        "labels": LABELS,
        "healthy_threshold": HEALTHY_THRESHOLD,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "db": db_engine is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
