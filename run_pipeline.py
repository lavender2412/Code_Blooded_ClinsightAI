# clinsight_pipeline.py
# ✅ LDA topic probabilities + TF-IDF (unigrams + bigrams) combined for better rating prediction
# ✅ Keeps your theme-level + per-review explainability using ONLY the topic-part coefficients

import os
import re
import json
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from scipy.sparse import hstack, csr_matrix


# ==============================
# CONFIG
# ==============================
DATA_PATH = "hospital.csv"   # update path if needed
TEXT_COL = "Feedback"
RATING_COL = "Ratings"

N_TOPICS = 5
TOP_WORDS = 12

PROB_THRESHOLD = 0.30   # for "topic present" frequency/recurrence
BOOTSTRAP_B = 200       # robustness for coefficient CI

# Task 4 config
TOP_N_ROADMAP = 5       # how many top themes to show in roadmap
USE_GPT = False         # set True if you want to call OpenAI API (needs OPENAI_API_KEY)
OPENAI_MODEL = "gpt-4o-mini"  # example; choose any available model


# ==============================
# 1) Setup NLTK
# ==============================
nltk.download("stopwords")
nltk.download("wordnet")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ==============================
# 2) Preprocessing
# ==============================
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [LEMMATIZER.lemmatize(w) for w in words if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)


# ==============================
# 3) LDA Utilities
# ==============================
def display_topics(model, feature_names, no_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics.append(top_words)
        print(f"\nTopic {topic_idx+1}:")
        print(" | ".join(top_words))
    return topics


# ==============================
# 4) Task 4: Recommendation + KPI Catalog (rule-based)
# ==============================
def action_catalog():
    return {
        "Wait Time & Operational Efficiency": {
            "effort": "High Effort",
            "recommendation": "Optimize appointment scheduling and triage workflow; add capacity buffers during peak hours.",
            "kpis": [
                {"kpi": "Avg waiting time (minutes)", "target": "< 15"},
                {"kpi": "On-time appointment rate (%)", "target": "> 85%"},
                {"kpi": "Patients leaving without being seen (%)", "target": "< 2%"},
            ],
        },
        "Overall Service & Facility Cleanliness": {
            "effort": "Quick Win",
            "recommendation": "Strengthen housekeeping SLA and hourly rounding; implement rapid-response cleaning for high-traffic areas.",
            "kpis": [
                {"kpi": "Cleanliness complaints per 100 visits", "target": "< 2"},
                {"kpi": "Room turnaround time (minutes)", "target": "< 30"},
                {"kpi": "Hygiene audit score (%)", "target": "> 95%"},
            ],
        },
        "Emergency Service Failures": {
            "effort": "High Effort",
            "recommendation": "Improve ER staffing and triage; standardize escalation protocols; implement real-time queue visibility.",
            "kpis": [
                {"kpi": "Door-to-provider time (minutes)", "target": "< 20"},
                {"kpi": "ER complaint rate per 100 visits", "target": "< 3"},
                {"kpi": "Escalations resolved within SLA (%)", "target": "> 90%"},
            ],
        },
        "Clinical Care & Treatment Quality": {
            "effort": "High Effort",
            "recommendation": "Standardize clinical pathways; introduce peer review + checklist adherence; increase follow-up quality checks.",
            "kpis": [
                {"kpi": "Clinical complaints per 100 visits", "target": "< 2"},
                {"kpi": "Follow-up completion rate (%)", "target": "> 80%"},
                {"kpi": "Readmission rate (if available)", "target": "Decrease MoM"},
            ],
        },
        "Consultation & Positive Patient Experience": {
            "effort": "Quick Win",
            "recommendation": "Improve communication scripts and consultation transparency; provide discharge instructions and next-steps summary.",
            "kpis": [
                {"kpi": "Communication complaints per 100 visits", "target": "< 2"},
                {"kpi": "Patient satisfaction (post-visit survey)", "target": "> 4.2/5"},
                {"kpi": "Instruction clarity score (%)", "target": "> 90%"},
            ],
        },
    }


# ==============================
# 5) Optional GPT — only if USE_GPT=True
# ==============================
def gpt_recommendation(theme_name: str, review_text: str, risk_level: str, kpis: list) -> dict:
    from openai import OpenAI
    client = OpenAI()

    prompt = f"""
You are an operations consultant for a hospital group.
Given a patient review and its detected theme, produce:
1) a short recommendation (1-2 sentences),
2) quick_wins (2 bullets),
3) high_effort_fixes (2 bullets),
4) KPIs to track (use the provided KPIs; do not invent new ones),
Return strict JSON with keys: recommendation, quick_wins, high_effort_fixes, kpis.

Theme: {theme_name}
Risk level: {risk_level}
Review: {review_text}
KPIs: {json.dumps(kpis)}
"""

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        text={"format": {"type": "json_object"}}
    )
    out = getattr(resp, "output_text", None)
    if out is None:
        out = resp.output[0].content[0].text
    return json.loads(out)


# ==============================
# MAIN
# ==============================
def main():
    # ------------------------------
    # Load data
    # ------------------------------
    df = pd.read_csv(DATA_PATH)
    df = df[[TEXT_COL, RATING_COL]].dropna()
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
    df = df.dropna(subset=[RATING_COL])
    df[RATING_COL] = df[RATING_COL].astype(float)

    # ------------------------------
    # Clean text
    # ------------------------------
    df["clean_text"] = df[TEXT_COL].apply(preprocess)

    # ------------------------------
    # LDA Vectorize (Count)
    # ------------------------------
    count_vectorizer = CountVectorizer(
        max_df=0.90,
        min_df=5,
        stop_words="english"
    )
    dtm = count_vectorizer.fit_transform(df["clean_text"])

    # ------------------------------
    # LDA
    # ------------------------------
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42,
        learning_method="batch"
    )
    lda.fit(dtm)

    feature_names = count_vectorizer.get_feature_names_out()
    topic_words = display_topics(lda, feature_names, no_top_words=TOP_WORDS)

    # ------------------------------
    # Topic probabilities per review (LDA features)
    # ------------------------------
    topic_prob = lda.transform(dtm)  # (n, K)

    for k in range(N_TOPICS):
        df[f"topic_prob_{k+1}"] = topic_prob[:, k]

    df["dominant_topic"] = topic_prob.argmax(axis=1) + 1  # 1..K

    # ------------------------------
    # Manual topic labels
    # ------------------------------
    topic_labels = {
        1: "Clinical Care & Treatment Quality",
        2: "Overall Service & Facility Cleanliness",
        3: "Emergency Service Failures",
        4: "Consultation & Positive Patient Experience",
        5: "Wait Time & Operational Efficiency",
    }
    df["theme_label"] = df["dominant_topic"].map(topic_labels)

    # ------------------------------
    # Theme frequency distribution
    # ------------------------------
    theme_counts = df["dominant_topic"].value_counts().sort_index()
    theme_freq_dom = (theme_counts / len(df)).rename("dominant_topic_frequency")

    # ------------------------------
    # Task 3 recurrence: topic present + frequency
    # ------------------------------
    present = (topic_prob > PROB_THRESHOLD).astype(int)
    present_freq = present.mean(axis=0)

    # ------------------------------
    # TF-IDF (unigrams + bigrams)
    # ------------------------------
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95
    )
    X_tfidf = tfidf_vectorizer.fit_transform(df["clean_text"])  # sparse

    # Combine: [TF-IDF | Topic probabilities]
    X_topics_sparse = csr_matrix(topic_prob)  # sparse (n, K)
    X = hstack([X_tfidf, X_topics_sparse]).tocsr()
    y = df[RATING_COL].values

    # ------------------------------
    # Task 2: Regression (TF-IDF + Topics -> rating)
    # IMPORTANT: with_mean=False because X is sparse
    # ------------------------------
    from sklearn.linear_model import Ridge

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("reg", Ridge(alpha=5967.90))
    ])

    model.fit(X, y)

    intercept = model.named_steps["reg"].intercept_
    full_coefs = model.named_steps["reg"].coef_

    # Topic-part coefficients = last K coefficients (because we stacked topics at the end)
    topic_coefs = full_coefs[-N_TOPICS:]

    # Robustness (CV)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")
    cv_rmse = -cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")

    # ------------------------------
    # Per-review LOCAL contributions (topic-only, dashboard explainability)
    # contrib_{i,k} = topic_prob_{i,k} * topic_coef_k
    # ------------------------------
    contributions = topic_prob * topic_coefs  # (n, K)
    for k in range(N_TOPICS):
        df[f"contrib_{k+1}"] = contributions[:, k]

    df["predicted_rating"] = model.predict(X)
    df["residual"] = df[RATING_COL] - df["predicted_rating"]

    contrib_cols = [f"contrib_{k+1}" for k in range(N_TOPICS)]
    abs_contrib = df[contrib_cols].abs()

    df["dominant_contributor_topic"] = abs_contrib.idxmax(axis=1).str.replace("contrib_", "").astype(int)
    df["dominant_contributor_theme"] = df["dominant_contributor_topic"].map(topic_labels)

    df["most_positive_topic"] = df[contrib_cols].idxmax(axis=1).str.replace("contrib_", "").astype(int)
    df["most_negative_topic"] = df[contrib_cols].idxmin(axis=1).str.replace("contrib_", "").astype(int)
    df["most_positive_theme"] = df["most_positive_topic"].map(topic_labels)
    df["most_negative_theme"] = df["most_negative_topic"].map(topic_labels)

    # ------------------------------
    # 1-star vs 5-star drivers (topic-prob only)
    # ------------------------------
    one_mask = (y == 1)
    five_mask = (y == 5)

    if one_mask.sum() > 0 and five_mask.sum() > 0:
        mean_1 = topic_prob[one_mask].mean(axis=0)
        mean_5 = topic_prob[five_mask].mean(axis=0)
        diff_1_5 = mean_1 - mean_5
    else:
        mean_1 = np.full(N_TOPICS, np.nan)
        mean_5 = np.full(N_TOPICS, np.nan)
        diff_1_5 = np.full(N_TOPICS, np.nan)

    # ------------------------------
    # Bootstrap coefficient CI + confidence score (topic-part only)
    # ------------------------------
    rng = np.random.default_rng(42)
    boot_topic_coefs = np.zeros((BOOTSTRAP_B, N_TOPICS))

    for b in range(BOOTSTRAP_B):
        idx = rng.integers(0, len(y), size=len(y))
        model.fit(X[idx], y[idx])
        boot_full = model.named_steps["reg"].coef_
        boot_topic_coefs[b] = boot_full[-N_TOPICS:]

    coef_mean = boot_topic_coefs.mean(axis=0)
    coef_lo = np.percentile(boot_topic_coefs, 2.5, axis=0)
    coef_hi = np.percentile(boot_topic_coefs, 97.5, axis=0)

    stability = 1 - (boot_topic_coefs.std(axis=0) / (np.abs(coef_mean) + 1e-8))
    stability = np.clip(stability, 0, 1)

    # ------------------------------
    # Severity + Risk score per theme/topic (topic-part impact)
    # ------------------------------
    severity = present_freq * np.abs(topic_coefs)
    risk = severity * stability

    # ------------------------------
    # Build THEME-LEVEL output table
    # ------------------------------
    theme_level = pd.DataFrame({
        "topic_id": np.arange(1, N_TOPICS + 1),
        "theme_label": [topic_labels[i] for i in range(1, N_TOPICS + 1)],
        "top_words": [" | ".join(topic_words[i-1]) for i in range(1, N_TOPICS + 1)],

        "dominant_topic_frequency": [theme_freq_dom.get(i, 0.0) for i in range(1, N_TOPICS + 1)],
        "present_frequency_(prob>thr)": present_freq,

        "impact_coefficient_topic_part": topic_coefs,
        "abs_impact_topic_part": np.abs(topic_coefs),

        "avg_in_1star": mean_1,
        "avg_in_5star": mean_5,
        "diff_(1star-5star)": diff_1_5,

        "coef_ci_2.5%": coef_lo,
        "coef_ci_97.5%": coef_hi,
        "confidence_stability": stability,

        "severity_score": severity,
        "risk_score": risk
    }).sort_values("risk_score", ascending=False)

    theme_level["cv_r2_mean"] = cv_r2.mean()
    theme_level["cv_r2_std"] = cv_r2.std()
    theme_level["cv_rmse_mean"] = cv_rmse.mean()
    theme_level["cv_rmse_std"] = cv_rmse.std()
    theme_level["intercept"] = intercept
    theme_level["tfidf_num_features"] = X_tfidf.shape[1]

    # Recurring/Systemic classification
    def classify(freq_present, abs_impact):
        if freq_present < 0.05:
            return "Isolated"
        if freq_present >= 0.20 and abs_impact >= np.median(np.abs(topic_coefs)):
            return "Systemic"
        return "Recurring"

    theme_level["issue_class"] = [
        classify(f, a) for f, a in zip(theme_level["present_frequency_(prob>thr)"], theme_level["abs_impact_topic_part"])
    ]

    # ==============================
    # TASK 3 output (recurring/systemic)
    # ==============================
    for k in range(N_TOPICS):
        df[f"topic_present_{k+1}"] = present[:, k]

    freq_df = pd.DataFrame({
        "topic_id": np.arange(1, N_TOPICS + 1),
        "theme_label": [topic_labels[i] for i in range(1, N_TOPICS + 1)],
        "reviews_with_topic": present.sum(axis=0).astype(int),
        "present_frequency": present_freq,
        "impact_coefficient_topic_part": topic_coefs,
        "abs_impact_topic_part": np.abs(topic_coefs),
        "confidence_stability": stability,
    })

    freq_df["risk_score_simple"] = freq_df["present_frequency"] * freq_df["abs_impact_topic_part"]
    freq_df["risk_score_confidence"] = freq_df["risk_score_simple"] * freq_df["confidence_stability"]
    freq_df["issue_class"] = [
        classify(f, a) for f, a in zip(freq_df["present_frequency"], freq_df["abs_impact_topic_part"])
    ]
    freq_df = freq_df.sort_values("risk_score_simple", ascending=False)

    # ==============================
    # TASK 4: Business Action Roadmap
    # ==============================
    catalog = action_catalog()

    roadmap_rows = []
    for _, row in freq_df.iterrows():
        theme = row["theme_label"]
        pack = catalog.get(theme, None)
        if pack is None:
            pack = {
                "effort": "Quick Win",
                "recommendation": "Investigate this theme and update SOPs; add monitoring and feedback loop.",
                "kpis": [{"kpi": "Complaints per 100 visits", "target": "< 3"}],
            }

        roadmap_rows.append({
            "priority_rank": None,
            "theme_label": theme,
            "issue_class": row["issue_class"],
            "risk_score": float(row["risk_score_confidence"]),
            "effort": pack["effort"],
            "recommendation": pack["recommendation"],
            "kpis": json.dumps(pack["kpis"]),
        })

    roadmap_df = pd.DataFrame(roadmap_rows).sort_values("risk_score", ascending=False).reset_index(drop=True)
    roadmap_df["priority_rank"] = np.arange(1, len(roadmap_df) + 1)
    roadmap_df["bucket"] = roadmap_df["effort"].apply(lambda x: "Quick Wins" if "Quick" in x else "High Effort")

    def risk_level_from_score(score: float) -> str:
        if np.isnan(score):
            return "Medium"
        q75 = roadmap_df["risk_score"].quantile(0.75)
        q35 = roadmap_df["risk_score"].quantile(0.35)
        if score >= q75:
            return "High"
        if score <= q35:
            return "Low"
        return "Medium"

    theme_to_risk = dict(zip(roadmap_df["theme_label"], roadmap_df["risk_score"]))
    theme_to_effort = dict(zip(roadmap_df["theme_label"], roadmap_df["effort"]))
    theme_to_rec = dict(zip(roadmap_df["theme_label"], roadmap_df["recommendation"]))
    theme_to_kpis = {r["theme_label"]: json.loads(r["kpis"]) for _, r in roadmap_df.iterrows()}
    theme_to_priority = dict(zip(roadmap_df["theme_label"], roadmap_df["priority_rank"]))

    df["review_theme_for_action"] = df["dominant_contributor_theme"]
    df["theme_risk_score"] = df["review_theme_for_action"].map(theme_to_risk)
    df["theme_priority_rank"] = df["review_theme_for_action"].map(theme_to_priority)
    df["effort_estimate"] = df["review_theme_for_action"].map(theme_to_effort)

    df["recommendation"] = df["review_theme_for_action"].map(theme_to_rec)
    df["kpi_pack"] = df["review_theme_for_action"].map(lambda t: json.dumps(theme_to_kpis.get(t, [])))

    if USE_GPT:
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️ USE_GPT=True but OPENAI_API_KEY not set. Falling back to rule-based recommendations.")
        else:
            refined = []
            for _, r in df.iterrows():
                theme = r["review_theme_for_action"]
                score = r["theme_risk_score"]
                risk_level = risk_level_from_score(score)
                kpis = json.loads(r["kpi_pack"])
                try:
                    out = gpt_recommendation(theme, r[TEXT_COL], risk_level, kpis)
                    refined.append(out)
                except Exception:
                    refined.append(None)

            rec_text, quick_wins, high_effort, kpis_json = [], [], [], []
            for out, (_, r) in zip(refined, df.iterrows()):
                if isinstance(out, dict):
                    rec_text.append(out.get("recommendation", r["recommendation"]))
                    quick_wins.append(json.dumps(out.get("quick_wins", [])))
                    high_effort.append(json.dumps(out.get("high_effort_fixes", [])))
                    kpis_json.append(json.dumps(out.get("kpis", json.loads(r["kpi_pack"]))))
                else:
                    rec_text.append(r["recommendation"])
                    quick_wins.append(json.dumps([]))
                    high_effort.append(json.dumps([]))
                    kpis_json.append(r["kpi_pack"])

            df["recommendation"] = rec_text
            df["quick_wins"] = quick_wins
            df["high_effort_fixes"] = high_effort
            df["kpi_pack"] = kpis_json
    else:
        df["quick_wins"] = json.dumps([])
        df["high_effort_fixes"] = json.dumps([])

    df["kpi_urgency_score"] = (
        (6 - df[RATING_COL]) * 0.6 +
        (df["theme_priority_rank"].fillna(TOP_N_ROADMAP + 1).rsub(TOP_N_ROADMAP + 1).clip(lower=0) * 0.4)
    )

    # ------------------------------
    # Review-level output
    # ------------------------------
    keep_cols = [
        TEXT_COL, RATING_COL, "clean_text",
        "dominant_topic", "theme_label",
        "predicted_rating", "residual",
        "dominant_contributor_topic", "dominant_contributor_theme",
        "most_positive_theme", "most_negative_theme",
    ]
    keep_cols += [f"topic_prob_{k+1}" for k in range(N_TOPICS)]
    keep_cols += [f"contrib_{k+1}" for k in range(N_TOPICS)]
    keep_cols += [f"topic_present_{k+1}" for k in range(N_TOPICS)]
    keep_cols += [
        "review_theme_for_action", "theme_risk_score", "theme_priority_rank",
        "effort_estimate", "recommendation", "kpi_pack",
        "quick_wins", "high_effort_fixes", "kpi_urgency_score"
    ]

    review_level = df[keep_cols].copy()

    # ------------------------------
    # Export
    # ------------------------------
    review_level.to_csv("review_level_outputs.csv", index=False)
    theme_level.to_csv("theme_level_outputs.csv", index=False)
    freq_df.to_csv("task3_recurring_systemic.csv", index=False)
    roadmap_df.to_csv("task4_action_roadmap.csv", index=False)

    print("\n✅ Saved: review_level_outputs.csv")
    print("✅ Saved: theme_level_outputs.csv")
    print("✅ Saved: task3_recurring_systemic.csv")
    print("✅ Saved: task4_action_roadmap.csv")

    print("\nModel robustness:")
    print(f"CV R^2:   mean={cv_r2.mean():.3f}, std={cv_r2.std():.3f}")
    print(f"CV RMSE:  mean={cv_rmse.mean():.3f}, std={cv_rmse.std():.3f}")
    print("\nTop roadmap items:\n", roadmap_df[["priority_rank", "theme_label", "risk_score", "effort"]].head(5))


if __name__ == "__main__":
    main()