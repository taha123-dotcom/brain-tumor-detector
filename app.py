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

db_engine = None
db_error_msg = None


def init_db():
    global db_engine, db_error_msg
    raw_url = os.environ.get("DATABASE_URL", "")
    if not raw_url:
        db_error_msg = "DATABASE_URL env var not set"
        print(f"DB: {db_error_msg}")
        return

    url = raw_url.strip()
    url = url.replace("postgres://", "postgresql://", 1)

    for fix in [
        ("?channel_binding=require", "?"),
        ("&channel_binding=require", ""),
        ("?channel_binding=require&", "?"),
        ("&channel_binding=require&", "&"),
    ]:
        url = url.replace(fix[0], fix[1])

    if url.endswith("&"):
        url = url[:-1]
    if url.endswith("?"):
        url = url[:-1]

    print(f"DB: connecting to {url.split('@')[-1] if '@' in url else url}...")

    try:
        import psycopg2
        from sqlalchemy import create_engine, text

        engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={"sslmode": "require", "connect_timeout": 10},
        )

        with engine.connect() as conn:
            r = conn.execute(text("SELECT 1"))
            print(f"DB: test query = {r.scalar()}")

        db_engine = engine
        db_error_msg = None

        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id VARCHAR(36) PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    prediction VARCHAR(50) NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    probabilities TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
        print("DB: table ready")

    except ImportError as e:
        db_error_msg = f"Missing driver: {e}"
        print(f"DB: {db_error_msg}")
    except Exception as e:
        db_error_msg = f"{type(e).__name__}: {e}"
        print(f"DB: {db_error_msg}")
        traceback.print_exc()


def db_save(filename, prediction, confidence, probabilities):
    global db_engine
    if not db_engine:
        return False, "no engine"
    try:
        from sqlalchemy import text
        pred_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()
        with db_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO predictions (id, filename, prediction, confidence, probabilities, created_at) "
                "VALUES (:id, :filename, :prediction, :confidence, :probabilities, :created_at)"
            ), {
                "id": pred_id,
                "filename": filename,
                "prediction": prediction,
                "confidence": float(confidence),
                "probabilities": json.dumps(probabilities),
                "created_at": now,
            })
        print(f"DB: saved {pred_id}")
        return True, None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"DB save failed: {err}")
        traceback.print_exc()
        return False, err


def db_history(limit=50):
    global db_engine
    if not db_engine:
        return []
    try:
        from sqlalchemy import text
        with db_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT id, filename, prediction, confidence, probabilities, created_at "
                "FROM predictions ORDER BY created_at DESC LIMIT :lim"
            ), {"lim": limit}).fetchall()
            results = []
            for row in rows:
                probs = row[4]
                if isinstance(probs, str):
                    try:
                        probs = json.loads(probs)
                    except Exception:
                        probs = {}
                results.append({
                    "id": row[0],
                    "filename": row[1],
                    "prediction": row[2],
                    "confidence": round(float(row[3]) * 100, 2),
                    "probabilities": {k: round(v * 100, 2) for k, v in probs.items()} if isinstance(probs, dict) else {},
                    "created_at": str(row[5]),
                })
            return results
    except Exception as e:
        print(f"DB history failed: {e}")
        return []


def db_stats():
    global db_engine
    total = 0
    dist = {}
    if db_engine:
        try:
            from sqlalchemy import text
            with db_engine.connect() as conn:
                total = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar() or 0
                rows = conn.execute(text(
                    "SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction"
                )).fetchall()
                dist = {r[0]: r[1] for r in rows}
        except Exception as e:
            print(f"DB stats failed: {e}")
    return total, dist


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
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]


load_model()
init_db()


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

    saved, db_err = db_save(file.filename, pred_label, confidence, prob_dict)

    return jsonify({
        "prediction": pred_label,
        "confidence": round(confidence * 100, 2),
        "probabilities": {k: round(v * 100, 2) for k, v in prob_dict.items()},
        "db_saved": saved,
        "db_error": db_err,
    })


@app.route("/api/history")
def history():
    return jsonify(db_history())


@app.route("/api/stats")
def stats():
    total, dist = db_stats()
    return jsonify({
        "total_predictions": total,
        "class_distribution": dist,
        "model_accuracy": METRICS.get("test_accuracy"),
        "classes": LABELS,
    })


@app.route("/api/debug")
def debug():
    raw_url = os.environ.get("DATABASE_URL", "")
    return jsonify({
        "db_engine_alive": db_engine is not None,
        "db_error": db_error_msg,
        "database_url_set": bool(raw_url),
        "database_url_preview": raw_url[:40] + "..." if len(raw_url) > 40 else raw_url,
        "model_loaded": session is not None,
        "labels": LABELS,
    })


@app.route("/api/test-db")
def test_db():
    if not db_engine:
        return jsonify({"ok": False, "error": db_error_msg})
    try:
        from sqlalchemy import text
        with db_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
        return jsonify({"ok": True, "prediction_count": count})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "db": db_engine is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
