import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import sentiment  # noqa: E402
from app import app  # noqa: E402

POS = "A brilliant, moving film with a wonderful cast. I loved it."
NEG = "Terrible, boring and a complete waste of time."


def test_predict_labels():
    assert sentiment.predict(POS)["label"] == "positive"
    assert sentiment.predict(NEG)["label"] == "negative"


def test_negation_is_handled():
    assert sentiment.predict("This is not a good movie.")["label"] == "negative"


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        sentiment.predict("   ")


def test_api_ok_and_errors():
    c = app.test_client()
    r = c.post("/api/predict", json={"review": POS})
    assert r.status_code == 200 and r.get_json()["label"] == "positive"
    assert c.post("/api/predict", json={}).status_code == 400
    assert c.post("/api/predict", json={"review": "x" * 6000}).status_code == 413
    assert c.get("/health").get_json() == {"status": "ok"}
    page = c.get("/")
    assert page.status_code == 200 and b'data-api="/api/predict"' in page.data
    assert c.get("/model.json").status_code == 200


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_model_matches_python():
    texts = [POS, NEG, "Not bad at all, actually quite good!", "meh"]
    out = subprocess.run(["node", os.path.join(ROOT, "tests", "parity.js"), json.dumps(texts)],
                         capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    for t, p_js in zip(texts, js):
        assert abs(sentiment.predict(t)["prob_positive"] - p_js) < 1e-3, t
