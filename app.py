"""Flask app: web UI + JSON API for the sentiment model.

Run locally:   python app.py            (http://localhost:5000)
Production:    gunicorn app:app
API:           POST /api/predict  {"review": "..."}  ->  {"label": "positive", "confidence": 0.93, ...}
import os

from flask import Flask, jsonify, request


HERE = os.path.dirname(os.path.abspath(__file__))
MAX_CHARS = 5000

app = Flask(__name__, static_folder=os.path.join(HERE, "docs"), static_url_path="")


def index():
    # Same page as the static GitHub Pages demo, but told to call this server's API.
    with open(os.path.join(HERE, "docs", "index.html"), encoding="utf-8") as f:
        html = f.read()
    return html.replace('data-api=""', 'data-api="/api/predict"', 1)


@app.get("/health")
def health():
    sentiment.get_model()
    return jsonify(status="ok")


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    review = data.get("review", "")
    if not isinstance(review, str) or not review.strip():
        return jsonify(error="Send JSON like {\"review\": \"text\"}"), 400
    if len(review) > MAX_CHARS:
        return jsonify(error=f"review too long (max {MAX_CHARS} characters)"), 413
    return jsonify(sentiment.predict(review))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
