# run_pipeline.py
# Run once locally: python3 run_pipeline.py
# Generates: review_level_outputs.csv, theme_level_outputs.csv, task3_recurring_systemic.csv

# ── SSL fix for Mac (must be before any nltk imports) ────────────────────────
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import re
import numpy as np
import pandas as pd
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = "hospital.csv"
TEXT_COL       = "Feedback"
RATING_COL     = "Ratings"
N_TOPICS       = 5
TOP_WORDS      = 12
PROB_THRESHOLD = 0.30
BOOTSTRAP_B    = 200

TOPIC_LABELS = {
    1: "Clinical Care & Treatment Quality",
    2: "Overall Service & Facility Cleanliness",
    3: "Emergency Service Failures",
    4: "Consultation & Positive Patient Experience",
    5: "Wait Time & Operational Efficiency",
}

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [LEMMATIZER.lemmatize(w) for w in words if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)

def display_topics(model, feature_names, no_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics.append(top_words)
        print(f"\nTopic {topic_idx+1}:")
        print(" | ".join(top_words))
    return topics


def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = df[[TEXT_COL, RATING_COL]].dropna()
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
    df = df.dropna(subset=[RATING_COL]).reset_index(drop=True)
    df[RATING_COL] = df[RATING_COL].astype(float)

    # ── Preprocess ────────────────────────────────────────────────────────────
    print("Preprocessing text...")
    df["clean_text"] = df[TEXT_COL].apply(preprocess)

    # ── Vectorize + LDA ───────────────────────────────────────────────────────
    print("Running LDA...")
    vectorizer = CountVectorizer(max_df=0.90, min_df=5, stop_words="english")
    dtm = vectorizer.fit_transform(df["clean_text"])

    lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42, learning_method="batch")
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()
    topic_words   = display_topics(lda, feature_names, no_top_words=TOP_WORDS)

    topic_prob = lda.transform(dtm)
    for k in range(N_TOPICS):
        df[f"topic_prob_{k+1}"] = topic_prob[:, k]

    df["dominant_topic"] = topic_prob.argmax(axis=1) + 1
    df["theme_label"]    = df["dominant_topic"].map(TOPIC_LABELS)

    # ── Frequency ─────────────────────────────────────────────────────────────
    theme_counts   = df["dominant_topic"].value_counts().sort_index()
    theme_freq_dom = (theme_counts / len(df)).rename("dominant_topic_frequency")
    present        = (topic_prob > PROB_THRESHOLD).astype(int)
    present_freq   = present.mean(axis=0)

    # ── Regression ────────────────────────────────────────────────────────────
    print("Running regression + bootstrap (this takes ~30 seconds)...")
    X = topic_prob
    y = df[RATING_COL].values

    model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    model.fit(X, y)
    intercept = model.named_steps["reg"].intercept_
    coefs     = model.named_steps["reg"].coef_

    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2   = cross_val_score(model, X, y, cv=kf, scoring="r2")
    cv_rmse = -cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")

    # ── Per-review contributions ───────────────────────────────────────────────
    contributions = X * coefs
    contrib_cols  = []
    for k in range(N_TOPICS):
        col = f"contrib_{k+1}"
        df[col] = contributions[:, k]
        contrib_cols.append(col)

    df["predicted_rating"] = model.predict(X)
    df["residual"]         = df[RATING_COL] - df["predicted_rating"]

    abs_contrib = df[contrib_cols].abs()
    df["dominant_contributor_topic"] = abs_contrib.idxmax(axis=1).str.replace("contrib_", "").astype(int)
    df["dominant_contributor_theme"] = df["dominant_contributor_topic"].map(TOPIC_LABELS)
    df["most_positive_topic"]        = df[contrib_cols].idxmax(axis=1).str.replace("contrib_", "").astype(int)
    df["most_negative_topic"]        = df[contrib_cols].idxmin(axis=1).str.replace("contrib_", "").astype(int)
    df["most_positive_theme"]        = df["most_positive_topic"].map(TOPIC_LABELS)
    df["most_negative_theme"]        = df["most_negative_topic"].map(TOPIC_LABELS)

    # ── 1-star vs 5-star ──────────────────────────────────────────────────────
    one_mask  = (y == 1)
    five_mask = (y == 5)
    if one_mask.sum() > 0 and five_mask.sum() > 0:
        mean_1   = X[one_mask].mean(axis=0)
        mean_5   = X[five_mask].mean(axis=0)
        diff_1_5 = mean_1 - mean_5
    else:
        mean_1 = mean_5 = diff_1_5 = np.full(N_TOPICS, np.nan)

    # ── Bootstrap CI ──────────────────────────────────────────────────────────
    rng        = np.random.default_rng(42)
    boot_coefs = np.zeros((BOOTSTRAP_B, N_TOPICS))
    for b in range(BOOTSTRAP_B):
        idx = rng.integers(0, len(y), size=len(y))
        model.fit(X[idx], y[idx])
        boot_coefs[b] = model.named_steps["reg"].coef_

    coef_mean = boot_coefs.mean(axis=0)
    coef_lo   = np.percentile(boot_coefs, 2.5,  axis=0)
    coef_hi   = np.percentile(boot_coefs, 97.5, axis=0)
    stability = np.clip(1 - (boot_coefs.std(axis=0) / (np.abs(coef_mean) + 1e-8)), 0, 1)

    # ── Severity & Risk ───────────────────────────────────────────────────────
    severity = present_freq * np.abs(coefs)
    risk     = severity * stability

    # ── Systemic classification ───────────────────────────────────────────────
    def classify(freq, abs_impact):
        if freq < 0.05:
            return "Isolated"
        if freq >= 0.20 and abs_impact >= np.median(np.abs(coefs)):
            return "Systemic"
        return "Recurring"

    # ── Theme-level output ────────────────────────────────────────────────────
    print("Building output tables...")
    theme_level = pd.DataFrame({
        "topic_id":                      np.arange(1, N_TOPICS + 1),
        "theme_label":                   [TOPIC_LABELS[i] for i in range(1, N_TOPICS + 1)],
        "top_words":                     [" | ".join(topic_words[i-1]) for i in range(1, N_TOPICS + 1)],
        "dominant_topic_frequency":      [theme_freq_dom.get(i, 0.0) for i in range(1, N_TOPICS + 1)],
        "present_frequency_(prob>thr)":  present_freq,
        "impact_coefficient":            coefs,
        "abs_impact":                    np.abs(coefs),
        "avg_in_1star":                  mean_1,
        "avg_in_5star":                  mean_5,
        "diff_(1star-5star)":            diff_1_5,
        "coef_ci_2.5%":                  coef_lo,
        "coef_ci_97.5%":                 coef_hi,
        "confidence_stability":          stability,
        "severity_score":                severity,
        "risk_score":                    risk,
        "cv_r2_mean":                    cv_r2.mean(),
        "cv_r2_std":                     cv_r2.std(),
        "cv_rmse_mean":                  cv_rmse.mean(),
        "cv_rmse_std":                   cv_rmse.std(),
        "intercept":                     intercept,
    })
    theme_level["issue_class"] = [
        classify(f, a) for f, a in zip(
            theme_level["present_frequency_(prob>thr)"],
            theme_level["abs_impact"]
        )
    ]
    theme_level = theme_level.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # ── Task 3: Recurring & Systemic table ───────────────────────────────────
    for k in range(N_TOPICS):
        df[f"topic_present_{k+1}"] = present[:, k]

    freq_df = pd.DataFrame({
        "topic_id":              np.arange(1, N_TOPICS + 1),
        "theme_label":           [TOPIC_LABELS[i] for i in range(1, N_TOPICS + 1)],
        "reviews_with_topic":    present.sum(axis=0).astype(int),
        "present_frequency":     present_freq,
        "impact_coefficient":    coefs,
        "abs_impact":            np.abs(coefs),
        "risk_score_simple":     present_freq * np.abs(coefs),
        "confidence_stability":  stability,
        "risk_score_confidence": risk,
    })
    freq_df["issue_class"] = [
        classify(f, a) for f, a in zip(freq_df["present_frequency"], freq_df["abs_impact"])
    ]
    freq_df = freq_df.sort_values("risk_score_simple", ascending=False).reset_index(drop=True)

    # ── Review-level output ───────────────────────────────────────────────────
    keep_cols = [
        TEXT_COL, RATING_COL, "clean_text",
        "dominant_topic", "theme_label",
        "predicted_rating", "residual",
        "dominant_contributor_topic", "dominant_contributor_theme",
        "most_positive_theme", "most_negative_theme"
    ]
    keep_cols += [f"topic_prob_{k+1}"    for k in range(N_TOPICS)]
    keep_cols += [f"contrib_{k+1}"       for k in range(N_TOPICS)]
    keep_cols += [f"topic_present_{k+1}" for k in range(N_TOPICS)]
    review_level = df[keep_cols].copy()

    # ── Save ──────────────────────────────────────────────────────────────────
    review_level.to_csv("review_level_outputs.csv", index=False)
    theme_level.to_csv("theme_level_outputs.csv",   index=False)
    freq_df.to_csv("task3_recurring_systemic.csv",  index=False)

    print("\n✅ Saved: review_level_outputs.csv")
    print("✅ Saved: theme_level_outputs.csv")
    print("✅ Saved: task3_recurring_systemic.csv")
    print("\nTop systemic risks:")
    print(freq_df[["topic_id", "theme_label", "issue_class", "risk_score_simple"]].to_string(index=False))


if __name__ == "__main__":
    main()