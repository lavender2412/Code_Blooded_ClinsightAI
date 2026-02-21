# run_pipeline.py
# Run this ONCE locally to generate the two CSV files:
#   python run_pipeline.py
# Then commit theme_level_outputs.csv and review_level_outputs.csv to GitHub.

import numpy as np
import pandas as pd
import ssl
import nltk
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


from pipeline.preprocess  import clean_dataframe
from pipeline.topic_model import run_lda, TOPIC_LABELS, N_TOPICS
from pipeline.impact      import run_impact
from pipeline.risk        import run_risk

DATA_PATH  = "hospital.csv"
TEXT_COL   = "Feedback"
RATING_COL = "Ratings"

def main():
    # ── Step 1: Load & preprocess ─────────────────────────────────────────────
    print("Loading and preprocessing...")
    raw_df = pd.read_csv(DATA_PATH)
    df     = clean_dataframe(raw_df, text_col=TEXT_COL, rating_col=RATING_COL)

    # ── Step 2: Topic modelling ───────────────────────────────────────────────
    print("Running LDA...")
    df, topic_prob, topic_words, theme_counts, theme_freq, topic_similarity = run_lda(df)

    # ── Step 3: Regression + impact ──────────────────────────────────────────
    print("Running regression + bootstrap (this takes ~30 seconds)...")
    df, impact_df, coefs, stability, model_stats = run_impact(df, topic_prob, rating_col=RATING_COL)

    # ── Step 4: Risk & systemic classification ────────────────────────────────
    print("Computing risk scores...")
    risk_df = run_risk(topic_prob, coefs, stability)

    # ── Step 5: Build theme-level output table ────────────────────────────────
    # Merge impact + risk into one table
    theme_level = impact_df.merge(risk_df, on="theme_label", suffixes=("", "_risk"))

    # Add topic_id, top_words, dominant frequency
    theme_level["topic_id"] = theme_level["theme_label"].map(
        {v: k for k, v in TOPIC_LABELS.items()}
    )
    theme_level["top_words"] = theme_level["theme_label"].map(
        {label: " | ".join(words) for label, words in topic_words.items()}
    )
    dominant_freq = (theme_counts / len(df)).rename("dominant_topic_frequency").reset_index()
    dominant_freq.columns = ["theme_label", "dominant_topic_frequency"]
    theme_level = theme_level.merge(dominant_freq, on="theme_label", how="left")

    # Add model stats as columns
    theme_level["cv_r2_mean"]   = model_stats["cv_r2_mean"]
    theme_level["cv_r2_std"]    = model_stats["cv_r2_std"]
    theme_level["cv_rmse_mean"] = model_stats["cv_rmse_mean"]
    theme_level["cv_rmse_std"]  = model_stats["cv_rmse_std"]
    theme_level["intercept"]    = model_stats["intercept"]

    theme_level = theme_level.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # ── Step 6: Build review-level output table ───────────────────────────────
    keep_cols = [
        TEXT_COL, RATING_COL, "clean_text",
        "dominant_topic", "theme_label",
        "predicted_rating", "residual",
        "dominant_contributor_topic", "dominant_contributor_theme",
        "most_positive_theme", "most_negative_theme"
    ]
    keep_cols += [f"topic_prob_{k+1}" for k in range(N_TOPICS)]
    keep_cols += [f"contrib_{k+1}"    for k in range(N_TOPICS)]
    review_level = df[keep_cols].copy()

    # ── Step 7: Save ──────────────────────────────────────────────────────────
    review_level.to_csv("review_level_outputs.csv", index=False)
    theme_level.to_csv("theme_level_outputs.csv",   index=False)

    print("\n✅ Saved: review_level_outputs.csv")
    print("✅ Saved: theme_level_outputs.csv")
    print("\nTop theme risks:")
    print(theme_level[["topic_id", "theme_label", "risk_score", "issue_class"]].to_string(index=False))

if __name__ == "__main__":
    main()