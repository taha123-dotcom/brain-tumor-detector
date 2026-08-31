import os
import json
import uuid
import datetime
import traceback
from io import BytesIO
from urllib.parse import urlparse
from contextlib import contextmanager

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

DB_HOST = None
DB_PORT = None
DB_NAME = None
DB_USER = None
DB_PASS = None
db_ready = False
db_error = None

DEFAULT_DB_URL = "postgresql://neondb_owner:npg_h9AD2giVlyLG@ep-shiny-dream-aeio87c7-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"


def parse_db():
    global DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS, db_ready, db_error
    import time

    raw = os.environ.get("DATABASE_URL", "") or DEFAULT_DB_URL
    if not raw:
        db_error = "No DATABASE_URL"
        print(f"DB: {db_error}")
        return
    url = raw.replace("postgres://", "postgresql://", 1)
    for bad in ["?channel_binding=require", "&channel_binding=require",
                "?channel_binding=require&", "&channel_binding=require&"]:
        url = url.replace(bad, "")
    if url.endswith("?") or url.endswith("&"):
        url = url[:-1]
    try:
        p = urlparse(url)
        DB_HOST = p.hostname
        DB_PORT = p.port or 5432
        DB_NAME = p.path.lstrip("/")
        DB_USER = p.username
        DB_PASS = p.password
        print(f"DB: parsed {DB_HOST}:{DB_PORT}/{DB_NAME} as {DB_USER}")
    except Exception as e:
        db_error = f"Parse failed: {e}"
        print(f"DB: {db_error}")
        return

    for attempt in range(3):
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                user=DB_USER, password=DB_PASS,
                sslmode="require", connect_timeout=15,
            )
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SELECT 1")
            print(f"DB: connected, test={cur.fetchone()[0]}")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    probabilities TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cur.close()
            conn.close()
            db_ready = True
            db_error = None
            print("DB: table ready")
            return
        except ImportError:
            db_error = "psycopg2 not installed"
            print(f"DB: {db_error}")
            return
        except Exception as e:
            db_error = f"{type(e).__name__}: {e}"
            print(f"DB: attempt {attempt+1}/3 failed: {db_error}")
            if attempt < 2:
                time.sleep(3)
    traceback.print_exc()


@contextmanager
def db_conn():
    import psycopg2
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
        sslmode="require", connect_timeout=10,
    )
    try:
        yield conn
    finally:
        conn.close()


def db_save(filename, prediction, confidence, probabilities):
    if not db_ready:
        return False, db_error or "db not ready"
    try:
        with db_conn() as conn:
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO predictions (id, filename, prediction, confidence, probabilities, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (str(uuid.uuid4()), filename, prediction, float(confidence),
                 json.dumps(probabilities), datetime.datetime.utcnow().isoformat()),
            )
            cur.close()
        print(f"DB: saved {filename} -> {prediction}")
        return True, None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"DB save failed: {err}")
        traceback.print_exc()
        return False, err


def db_history(limit=50):
    if not db_ready:
        return []
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, filename, prediction, confidence, probabilities, created_at "
                "FROM predictions ORDER BY created_at DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
            cur.close()
            results = []
            for row in rows:
                probs = row[4]
                if isinstance(probs, str):
                    try:
                        probs = json.loads(probs)
                    except Exception:
                        probs = {}
                results.append({
                    "id": row[0], "filename": row[1], "prediction": row[2],
                    "confidence": round(float(row[3]) * 100, 2),
                    "probabilities": {k: round(v * 100, 2) for k, v in probs.items()} if isinstance(probs, dict) else {},
                    "created_at": str(row[5]),
                })
            return results
    except Exception as e:
        print(f"DB history failed: {e}")
        return []


def db_stats():
    if not db_ready:
        return 0, {}
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM predictions")
            total = cur.fetchone()[0] or 0
            cur.execute("SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction")
            dist = {r[0]: r[1] for r in cur.fetchall()}
            cur.close()
            return total, dist
    except Exception as e:
        print(f"DB stats failed: {e}")
        return 0, {}


def load_model():
    global session, LABELS, METRICS, IMG_SIZE, HEALTHY_THRESHOLD
    model_path = os.path.join(MODEL_DIR, "model.onnx")
    if not os.path.exists(model_path):
        for alt in [os.path.join(os.getcwd(), "models", "model.onnx"),
                     os.path.join(os.getcwd(), "brain_tumor_model.onnx")]:
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
        [0.229, 0.224, 0.225], dtype=np.float32)
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]


load_model()
parse_db()


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


@app.route("/health")
def health():
    return jsonify({"status": "ok", "db": db_ready, "error": db_error})


@app.route("/api/test-db")
def test_db():
    if not db_ready:
        return jsonify({"ok": False, "error": db_error, "host": DB_HOST})
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM predictions")
            count = cur.fetchone()[0]
            cur.close()
            return jsonify({"ok": True, "count": count})
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
