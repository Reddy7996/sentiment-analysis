"""
Train the movie-review sentiment model.

Pipeline: TF-IDF (unigrams + bigrams, sentiment-safe stop-words) -> Logistic Regression.
Hyper-parameters (C, weight of hand-written reviews) are grid-searched with 5-fold CV.

Data
----
* data/reviews.csv            5,000 template-generated reviews (see generate_sample_data.py)
* data/handwritten_reviews.csv 90 hand-written, natural-sounding reviews
    - split == "train": added to training (up-weighted so natural phrasing, e.g. negation, is learned)
    - split == "test" : NEVER trained on; the honest, realistic held-out evaluation

Outputs
-------
model/model.joblib        sklearn pipeline (used by app.py / predict.py)
model/metrics.json        all reported numbers (README is written from this file)
docs/model.json           the same model exported as JSON for the static GitHub Pages demo
visualizations/*.png      confusion matrix + top-word charts

Usage:  python train.py [--data path/to/your_reviews.csv]
        (a custom CSV needs columns: review, sentiment[0/1])
"""
import argparse
import json
import os
import subprocess
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
LABELS = ["Negative", "Positive"]
# English stop-words, but keep the words that flip or intensify sentiment ("not good", "no plot")
KEEP = {"not", "no", "nor", "never", "very", "too", "without", "nothing", "cannot", "against"}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS - KEEP)


def load_data(path):
    if not os.path.exists(path):
        print(f"{path} not found - generating it with generate_sample_data.py")
        subprocess.run([sys.executable, "generate_sample_data.py"], check=True)
    main = pd.read_csv(path)[["review", "sentiment"]]
    main = main.dropna().drop_duplicates()
    hand = pd.read_csv("data/handwritten_reviews.csv")
    return main, hand


def make_pipeline(C=1.0):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, token_pattern=r"[a-z']+",
                                  ngram_range=(1, 2), min_df=2, max_df=0.9,
                                  max_features=5000, stop_words=STOP_WORDS)),
        ("clf", LogisticRegression(C=C, max_iter=1000, random_state=SEED)),
    ])


def metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "n": int(len(y_true)),
    }


def export_json(pipe, path):
    """Export TF-IDF + logistic regression so the browser demo can run it in plain JS."""
    tfidf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    vocab = {t: int(i) for t, i in tfidf.vocabulary_.items()}
    with open(path, "w") as f:
        json.dump({
            "vocab": vocab,
            "idf": [round(float(x), 6) for x in tfidf.idf_],
            "coef": [round(float(x), 6) for x in clf.coef_[0]],
            "intercept": round(float(clf.intercept_[0]), 6),
            "stop_words": STOP_WORDS,
        }, f, separators=(",", ":"))


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_words(pipe, path, k=15):
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    neg, pos = order[:k], order[::-1][:k]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, idx, color, title in ((axes[0], neg, "#d9534f", "Most negative terms"),
                                  (axes[1], pos, "#2a9d8f", "Most positive terms")):
        ax.barh(range(k), coef[idx], color=color)
        ax.set_yticks(range(k), names[idx])
        ax.invert_yaxis()
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Logistic-regression coefficient")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_terms(pipe, k=10):
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    return {"positive": [str(names[i]) for i in order[::-1][:k]],
            "negative": [str(names[i]) for i in order[:k]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/reviews.csv")
    args = ap.parse_args()

    main_df, hand = load_data(args.data)
    hand_train = hand[hand.split == "train"]
    hand_test = hand[hand.split == "test"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        main_df.review, main_df.sentiment, test_size=0.2, random_state=SEED,
        stratify=main_df.sentiment)

    # ---- 1) Baseline: trained on the main dataset only --------------------------------
    base = make_pipeline().fit(X_tr, y_tr)
    res = {
        "dataset": {"main_rows": int(len(main_df)), "main_file": args.data,
                    "handwritten_train": int(len(hand_train)),
                    "handwritten_test": int(len(hand_test))},
        "baseline_main_only": {
            "on_main_test": metrics(y_te, base.predict(X_te)),
            "on_handwritten_test": metrics(hand_test.sentiment, base.predict(hand_test.review)),
        },
    }

    # ---- 2) Final model: main + up-weighted hand-written train, tuned -----------------
    def fit_weighted(C, weight, X_extra, y_extra):
        X = pd.concat([X_tr, X_extra], ignore_index=True)
        y = pd.concat([y_tr, y_extra], ignore_index=True)
        w = np.concatenate([np.ones(len(X_tr)), np.full(len(X_extra), weight)])
        return make_pipeline(C).fit(X, y, clf__sample_weight=w)

    # Tune (C, weight) by 5-fold CV over the *hand-written train* reviews only: the template
    # data is separable (100% for any setting), so it cannot tell good settings from bad ones.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for C in (0.1, 1, 10, 100):
        for weight in (5, 25, 50):
            accs = []
            for a, b in skf.split(hand_train.review, hand_train.sentiment):
                m = fit_weighted(C, weight, hand_train.review.iloc[a], hand_train.sentiment.iloc[a])
                accs.append(accuracy_score(hand_train.sentiment.iloc[b], m.predict(hand_train.review.iloc[b])))
            scores[(C, weight)] = float(np.mean(accs))
    (best_C, best_w), best_cv = max(scores.items(), key=lambda kv: (kv[1], -kv[0][0]))
    best = fit_weighted(best_C, best_w, hand_train.review, hand_train.sentiment)

    pred_main = best.predict(X_te)
    pred_hand = best.predict(hand_test.review)
    res["best_params"] = {"C": best_C, "handwritten_weight": best_w}
    res["handwritten_cv_accuracy"] = round(best_cv, 4)
    res["final_model"] = {
        "on_main_test": metrics(y_te, pred_main),
        "on_handwritten_test": metrics(hand_test.sentiment, pred_hand),
    }
    res["top_terms"] = top_terms(best)

    # ---- save everything ------------------------------------------------------------
    os.makedirs("model", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    joblib.dump(best, "model/model.joblib")
    export_json(best, "docs/model.json")
    for p in ("model/metrics.json", "docs/metrics.json"):
        with open(p, "w") as f:
            json.dump(res, f, indent=2)
    plot_confusion(confusion_matrix(hand_test.sentiment, pred_hand),
                   "Hand-written held-out reviews", "visualizations/confusion_matrix_handwritten.png")
    plot_confusion(confusion_matrix(y_te, pred_main),
                   "Template-generated test set", "visualizations/confusion_matrix_main.png")
    plot_top_words(best, "visualizations/top_words.png")

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
