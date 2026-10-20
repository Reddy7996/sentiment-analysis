"""Load the trained model and predict sentiment (with the words that drove the decision)."""
import os

import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.joblib")
_pipe = None


def get_model():
    global _pipe
    if _pipe is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("model/model.joblib not found - run `python train.py` first")
        _pipe = joblib.load(MODEL_PATH)
    return _pipe


def predict(text, top_k=6):
    """Return {label, confidence, prob_positive, top_words} for one review."""
    text = (text or "").strip()
    if not text:
        raise ValueError("review text is empty")
    pipe = get_model()
    tfidf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    x = tfidf.transform([text])
    p_pos = float(clf.predict_proba(x)[0, 1])
    names = tfidf.get_feature_names_out()
    idx = x.nonzero()[1]
    contrib = [(str(names[j]), float(x[0, j] * clf.coef_[0][j])) for j in idx]
    contrib.sort(key=lambda t: abs(t[1]), reverse=True)
    label = "positive" if p_pos >= 0.5 else "negative"
    return {
        "label": label,
        "confidence": round(max(p_pos, 1 - p_pos), 4),
        "prob_positive": round(p_pos, 4),
        "top_words": [{"word": w, "weight": round(c, 4)} for w, c in contrib[:top_k]],
        "known_words": int(len(idx)),
    }


if __name__ == "__main__":
    import sys
    review = " ".join(sys.argv[1:]) or "An unforgettable, beautifully acted film."
    print(predict(review))
