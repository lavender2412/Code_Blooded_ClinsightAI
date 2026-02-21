# run_pipeline.py
# Run once locally: python3 run_pipeline.py
# Generates: review_level_outputs.csv, theme_level_outputs.csv,
#            task3_recurring_systemic.csv, task4_action_roadmap.csv, clinsight_output.json

# ── SSL fix for Mac (must come before any nltk imports) ───────────────────────
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import re
import json
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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score


# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = "hospital.csv"
TEXT_COL       = "Feedback"
RATING_COL     = "Ratings"
N_TOPICS       = 5
TOP_WORDS      = 12
PROB_THRESHOLD = 0.30
BOOTSTRAP_B    = 200
TOP_N_ROADMAP  = 5
USE_GPT        = False
OPENAI_MODEL   = "gpt-4o-mini"

TOPIC_LABELS = {
    1: "Clinical Care & Treatment Quality",
    2: "Overall Service & Facility Cleanliness",
    3: "Emergency Service Failures",
    4: "Consultation & Positive Patient Experience",
    5: "Wait Time & Operational Efficiency",
}

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ── Preprocessing ─────────────────────────────────────────────────────────────
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


# ── Task 4: Action Catalog ────────────────────────────────────────────────────
def action_catalog():
    return {
        "Wait Time & Operational Efficiency": {
            "effort": "High Effort",
            "recommendation": "Optimize appointment scheduling and triage workflow; add capacity buffers during peak hours.",
            "kpis": [
                {"kpi": "Avg waiting time (minutes)",              "target": "< 15"},
                {"kpi": "On-time appointment rate (%)",            "target": "> 85%"},
                {"kpi": "Patients leaving without being seen (%)", "target": "< 2%"},
            ],
        },
        "Overall Service & Facility Cleanliness": {
            "effort": "Quick Win",
            "recommendation": "Strengthen housekeeping SLA and hourly rounding; implement rapid-response cleaning for high-traffic areas.",
            "kpis": [
                {"kpi": "Cleanliness complaints per 100 visits", "target": "< 2"},
                {"kpi": "Room turnaround time (minutes)",        "target": "< 30"},
                {"kpi": "Hygiene audit score (%)",              "target": "> 95%"},
            ],
        },
        "Emergency Service Failures": {
            "effort": "High Effort",
            "recommendation": "Improve ER staffing and triage; standardize escalation protocols; implement real-time queue visibility.",
            "kpis": [
                {"kpi": "Door-to-provider time (minutes)",      "target": "< 20"},
                {"kpi": "ER complaint rate per 100 visits",     "target": "< 3"},
                {"kpi": "Escalations resolved within SLA (%)", "target": "> 90%"},
            ],
        },
        "Clinical Care & Treatment Quality": {
            "effort": "High Effort",
            "recommendation": "Standardize clinical pathways; introduce peer review + checklist adherence; increase follow-up quality checks.",
            "kpis": [
                {"kpi": "Clinical complaints per 100 visits", "target": "< 2"},
                {"kpi": "Follow-up completion rate (%)",      "target": "> 80%"},
                {"kpi": "Readmission rate (if available)",    "target": "Decrease MoM"},
            ],
        },
        "Consultation & Positive Patient Experience": {
            "effort": "Quick Win",
            "recommendation": "Improve communication scripts and consultation transparency; provide discharge instructions and next-steps summary.",
            "kpis": [
                {"kpi": "Communication complaints per 100 visits",  "target": "< 2"},
                {"kpi": "Patient satisfaction (post-visit survey)", "target": "> 4.2/5"},
                {"kpi": "Instruction clarity score (%)",            "target": "> 90%"},
            ],
        },
    }


# ── Optional GPT ──────────────────────────────────────────────────────────────
def gpt_recommendation(theme_name, review_text, risk_level, kpis):
    from openai import OpenAI
    client = OpenAI()
    prompt = f"""
You are an operations consultant for a hospital group.
Given a patient review and its detected theme, produce:
1) a short recommendation (1-2 sentences),
2) quick_wins (2 bullets),
3) high_effort_fixes (2 bullets),
4) KPIs to track (use the provided KPIs; do not invent new ones).
Return strict JSON with keys: recommendation, quick_wins, high_effort_fixes, kpis.

Theme: {theme_name}
Risk level: {risk_level}
Review: {review_text}
KPIs: {json.dumps(kpis)}
"""
    resp = client.responses.create(model=OPENAI_MODEL, input=prompt,
                                   text={"format": {"type": "json_object"}})
    out = getattr(resp, "output_text", None)
    if out is None:
        out = resp.output[0].content[0].text
    return json.loads(out)


# ── JSON output generator ─────────────────────────────────────────────────────
def generate_json_output(theme_level, review_level, roadmap_df):
    overall_rating_mean = round(float(review_level[RATING_COL].mean()), 2)
    top_risk_themes     = theme_level.sort_values("risk_score", ascending=False)["theme_label"].head(2).tolist()

    clinic_summary = {
        "overall_rating_mean":  overall_rating_mean,
        "total_reviews":        int(len(review_level)),
        "avg_predicted_rating": round(float(review_level["predicted_rating"].mean()), 2),
        "primary_risk_themes":  top_risk_themes,
    }

    theme_analysis = []
    for _, row in theme_level.iterrows():
        label   = row["theme_label"]
        samples = (
            review_level[(review_level["theme_label"] == label) & (review_level[RATING_COL] <= 2)]
            ["Feedback"].dropna().head(3).tolist()
        )
        if not samples:
            samples = review_level[review_level["theme_label"] == label]["Feedback"].dropna().head(3).tolist()
        samples = [str(s)[:80] + "..." if len(str(s)) > 80 else str(s) for s in samples]
        theme_analysis.append({
            "theme":                label,
            "frequency_percentage": round(float(row["dominant_topic_frequency"]) * 100, 1),
            "rating_impact":        round(float(row["impact_coefficient"]), 3),
            "severity_score":       round(float(row["severity_score"]), 4),
            "risk_score":           round(float(row["risk_score"]), 4),
            "issue_class":          row["issue_class"],
            "confidence_stability": round(float(row["confidence_stability"]), 3),
            "evidence_samples":     samples,
        })

    improvement_roadmap = []
    for _, row in roadmap_df.iterrows():
        kpis = json.loads(row["kpis"]) if isinstance(row["kpis"], str) else []
        improvement_roadmap.append({
            "priority":             int(row["priority_rank"]),
            "theme":                row["theme_label"],
            "recommendation":       row["recommendation"],
            "effort":               row["effort"],
            "expected_rating_lift": f"+{round(float(row['risk_score']) * 0.5, 2)}",
            "confidence":           round(float(row["risk_score"]), 3),
            "issue_class":          row["issue_class"],
            "kpis":                 kpis,
        })

    return {
        "clinic_summary":      clinic_summary,
        "theme_analysis":      theme_analysis,
        "improvement_roadmap": improvement_roadmap,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = df[[TEXT_COL, RATING_COL]].dropna()
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
    df = df.dropna(subset=[RATING_COL]).reset_index(drop=True)
    df[RATING_COL] = df[RATING_COL].astype(float)

    # Preprocess
    print("Preprocessing...")
    df["clean_text"] = df[TEXT_COL].apply(preprocess)

    # Vectorize + LDA
    print("Running LDA...")
    vectorizer = CountVectorizer(max_df=0.90, min_df=5, stop_words="english")
    dtm        = vectorizer.fit_transform(df["clean_text"])
    lda        = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42, learning_method="batch")
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()
    topic_words   = display_topics(lda, feature_names, no_top_words=TOP_WORDS)

    topic_prob = lda.transform(dtm)
    for k in range(N_TOPICS):
        df[f"topic_prob_{k+1}"] = topic_prob[:, k]
    df["dominant_topic"] = topic_prob.argmax(axis=1) + 1
    df["theme_label"]    = df["dominant_topic"].map(TOPIC_LABELS)

    # Frequency
    theme_counts   = df["dominant_topic"].value_counts().sort_index()
    theme_freq_dom = (theme_counts / len(df)).rename("dominant_topic_frequency")
    present        = (topic_prob > PROB_THRESHOLD).astype(int)
    present_freq   = present.mean(axis=0)

    # Regression
    print("Running regression + bootstrap (~30s)...")
    X = topic_prob
    y = df[RATING_COL].values
    #model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    #model.fit(X, y)
    # Tests multiple alpha values automatically via cross-validation

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

    model = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("reg", Ridge(alpha=5967.90))
        ])    
    model.fit(X, y)


    intercept = model.named_steps["reg"].intercept_
    coefs     = model.named_steps["reg"].coef_

    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2   = cross_val_score(model, X, y, cv=kf, scoring="r2")
    cv_rmse = -cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")

    # Per-review contributions
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

    # 1-star vs 5-star
    one_mask  = (y == 1);  five_mask = (y == 5)
    if one_mask.sum() > 0 and five_mask.sum() > 0:
        mean_1 = X[one_mask].mean(axis=0);  mean_5 = X[five_mask].mean(axis=0)
        diff_1_5 = mean_1 - mean_5
    else:
        mean_1 = mean_5 = diff_1_5 = np.full(N_TOPICS, np.nan)

    # Bootstrap CI
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

    severity = present_freq * np.abs(coefs)
    risk     = severity * stability

    def classify(freq, abs_impact):
        if freq < 0.05:
            return "Isolated"
        if freq >= 0.20 and abs_impact >= np.median(np.abs(coefs)):
            return "Systemic"
        return "Recurring"

    # Theme-level table
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
        classify(f, a) for f, a in zip(theme_level["present_frequency_(prob>thr)"], theme_level["abs_impact"])
    ]
    theme_level = theme_level.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # Task 3 table
    for k in range(N_TOPICS):
        df[f"topic_present_{k+1}"] = present[:, k]

    freq_df = pd.DataFrame({
        "topic_id":             np.arange(1, N_TOPICS + 1),
        "theme_label":          [TOPIC_LABELS[i] for i in range(1, N_TOPICS + 1)],
        "reviews_with_topic":   present.sum(axis=0).astype(int),
        "present_frequency":    present_freq,
        "impact_coefficient":   coefs,
        "abs_impact":           np.abs(coefs),
        "confidence_stability": stability,
    })
    freq_df["risk_score_simple"]     = freq_df["present_frequency"] * freq_df["abs_impact"]
    freq_df["risk_score_confidence"] = freq_df["risk_score_simple"] * freq_df["confidence_stability"]
    freq_df["issue_class"] = [
        classify(f, a) for f, a in zip(freq_df["present_frequency"], freq_df["abs_impact"])
    ]
    freq_df = freq_df.sort_values("risk_score_simple", ascending=False).reset_index(drop=True)

    # Task 4: Roadmap
    print("Building roadmap...")
    catalog      = action_catalog()
    roadmap_rows = []
    for _, row in freq_df.iterrows():
        theme = row["theme_label"]
        pack  = catalog.get(theme, {
            "effort":         "Quick Win",
            "recommendation": "Investigate this theme and update SOPs; add monitoring and feedback loop.",
            "kpis":           [{"kpi": "Complaints per 100 visits", "target": "< 3"}],
        })
        roadmap_rows.append({
            "priority_rank":  None,
            "theme_label":    theme,
            "issue_class":    row["issue_class"],
            "risk_score":     float(row["risk_score_confidence"]),
            "effort":         pack["effort"],
            "recommendation": pack["recommendation"],
            "kpis":           json.dumps(pack["kpis"]),
        })

    roadmap_df = pd.DataFrame(roadmap_rows).sort_values("risk_score", ascending=False).reset_index(drop=True)
    roadmap_df["priority_rank"] = np.arange(1, len(roadmap_df) + 1)
    roadmap_df["bucket"]        = roadmap_df["effort"].apply(lambda x: "Quick Wins" if "Quick" in x else "High Effort")

    def risk_level_from_score(score):
        if np.isnan(score): return "Medium"
        q75 = roadmap_df["risk_score"].quantile(0.75)
        q35 = roadmap_df["risk_score"].quantile(0.35)
        return "High" if score >= q75 else ("Low" if score <= q35 else "Medium")

    # Per-review Task 4 columns
    theme_to_risk     = dict(zip(roadmap_df["theme_label"], roadmap_df["risk_score"]))
    theme_to_effort   = dict(zip(roadmap_df["theme_label"], roadmap_df["effort"]))
    theme_to_rec      = dict(zip(roadmap_df["theme_label"], roadmap_df["recommendation"]))
    theme_to_kpis     = {r["theme_label"]: json.loads(r["kpis"]) for _, r in roadmap_df.iterrows()}
    theme_to_priority = dict(zip(roadmap_df["theme_label"], roadmap_df["priority_rank"]))

    df["review_theme_for_action"] = df["dominant_contributor_theme"]
    df["theme_risk_score"]        = df["review_theme_for_action"].map(theme_to_risk)
    df["theme_priority_rank"]     = df["review_theme_for_action"].map(theme_to_priority)
    df["effort_estimate"]         = df["review_theme_for_action"].map(theme_to_effort)
    df["recommendation"]          = df["review_theme_for_action"].map(theme_to_rec)
    df["kpi_pack"]                = df["review_theme_for_action"].map(lambda t: json.dumps(theme_to_kpis.get(t, [])))

    if USE_GPT:
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️ USE_GPT=True but OPENAI_API_KEY not set. Falling back to rule-based.")
        else:
            refined = []
            for i, r in df.iterrows():
                theme      = r["review_theme_for_action"]
                risk_level = risk_level_from_score(r["theme_risk_score"])
                kpis       = json.loads(r["kpi_pack"])
                try:
                    out = gpt_recommendation(theme, r[TEXT_COL], risk_level, kpis)
                    refined.append(out)
                except Exception:
                    refined.append(None)

            rec_text = [];  quick_wins = [];  high_effort = [];  kpis_json = []
            for out, (_, r) in zip(refined, df.iterrows()):
                if isinstance(out, dict):
                    rec_text.append(out.get("recommendation", r["recommendation"]))
                    quick_wins.append(json.dumps(out.get("quick_wins", [])))
                    high_effort.append(json.dumps(out.get("high_effort_fixes", [])))
                    kpis_json.append(json.dumps(out.get("kpis", json.loads(r["kpi_pack"]))))
                else:
                    rec_text.append(r["recommendation"])
                    quick_wins.append(json.dumps([]));  high_effort.append(json.dumps([]))
                    kpis_json.append(r["kpi_pack"])

            df["recommendation"]    = rec_text
            df["quick_wins"]        = quick_wins
            df["high_effort_fixes"] = high_effort
            df["kpi_pack"]          = kpis_json
    else:
        df["quick_wins"]        = json.dumps([])
        df["high_effort_fixes"] = json.dumps([])

    df["kpi_urgency_score"] = (
        (6 - df[RATING_COL]) * 0.6 +
        (df["theme_priority_rank"].fillna(TOP_N_ROADMAP + 1).rsub(TOP_N_ROADMAP + 1).clip(lower=0) * 0.4)
    )

    # Review-level table
    keep_cols = [
        TEXT_COL, RATING_COL, "clean_text",
        "dominant_topic", "theme_label",
        "predicted_rating", "residual",
        "dominant_contributor_topic", "dominant_contributor_theme",
        "most_positive_theme", "most_negative_theme",
    ]
    keep_cols += [f"topic_prob_{k+1}"    for k in range(N_TOPICS)]
    keep_cols += [f"contrib_{k+1}"       for k in range(N_TOPICS)]
    keep_cols += [f"topic_present_{k+1}" for k in range(N_TOPICS)]
    keep_cols += [
        "review_theme_for_action", "theme_risk_score", "theme_priority_rank",
        "effort_estimate", "recommendation", "kpi_pack",
        "quick_wins", "high_effort_fixes", "kpi_urgency_score"
    ]
    review_level = df[keep_cols].copy()

    # Save CSVs
    review_level.to_csv("review_level_outputs.csv", index=False)
    theme_level.to_csv("theme_level_outputs.csv",   index=False)
    freq_df.to_csv("task3_recurring_systemic.csv",  index=False)
    roadmap_df.to_csv("task4_action_roadmap.csv",   index=False)

    print("\n✅ Saved: review_level_outputs.csv")
    print("✅ Saved: theme_level_outputs.csv")
    print("✅ Saved: task3_recurring_systemic.csv")
    print("✅ Saved: task4_action_roadmap.csv")

    # Save JSON
    output = generate_json_output(theme_level, review_level, roadmap_df)
    with open("clinsight_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print("✅ Saved: clinsight_output.json")

    print("\nTop roadmap items:")
    print(roadmap_df[["priority_rank", "theme_label", "risk_score", "effort"]].to_string(index=False))


if __name__ == "__main__":
    main()