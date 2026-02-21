import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from pipeline.topic_model import N_TOPICS, TOPIC_LABELS

PROB_THRESHOLD = 0.30
BOOTSTRAP_B    = 200

def run_impact(df, topic_prob, rating_col="Ratings"):
    """
    Fit linear regression (topic probs -> rating).
    Returns:
        df           — with predicted_rating, residual, contrib_k, contributor columns
        impact_df    — per-theme coefficient table
        coefs        — raw coefficient array
        stability    — bootstrap stability array
        model_stats  — dict with cv_r2, cv_rmse, intercept
    """
    X = topic_prob
    y = df[rating_col].values

    model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
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
    df["residual"]         = df[rating_col] - df["predicted_rating"]

    abs_contrib = df[contrib_cols].abs()
    df["dominant_contributor_topic"] = abs_contrib.idxmax(axis=1).str.replace("contrib_", "").astype(int)
    df["dominant_contributor_theme"] = df["dominant_contributor_topic"].map(TOPIC_LABELS)
    df["most_positive_topic"]        = df[contrib_cols].idxmax(axis=1).str.replace("contrib_", "").astype(int)
    df["most_negative_topic"]        = df[contrib_cols].idxmin(axis=1).str.replace("contrib_", "").astype(int)
    df["most_positive_theme"]        = df["most_positive_topic"].map(TOPIC_LABELS)
    df["most_negative_theme"]        = df["most_negative_topic"].map(TOPIC_LABELS)

    # 1-star vs 5-star
    one_mask  = (y == 1)
    five_mask = (y == 5)
    mean_1 = X[one_mask].mean(axis=0)  if one_mask.sum()  > 0 else np.full(N_TOPICS, np.nan)
    mean_5 = X[five_mask].mean(axis=0) if five_mask.sum() > 0 else np.full(N_TOPICS, np.nan)

    # Bootstrap CI
    rng        = np.random.default_rng(42)
    boot_coefs = np.zeros((BOOTSTRAP_B, N_TOPICS))
    for b in range(BOOTSTRAP_B):
        idx = rng.integers(0, len(y), size=len(y))
        model.fit(X[idx], y[idx])
        boot_coefs[b] = model.named_steps["reg"].coef_

    coef_lo   = np.percentile(boot_coefs, 2.5,  axis=0)
    coef_hi   = np.percentile(boot_coefs, 97.5, axis=0)
    stability = np.clip(
        1 - (boot_coefs.std(axis=0) / (np.abs(boot_coefs.mean(axis=0)) + 1e-8)),
        0, 1
    )

    impact_df = pd.DataFrame({
        "theme_label":          [TOPIC_LABELS[i] for i in range(1, N_TOPICS + 1)],
        "impact_coefficient":   coefs,
        "abs_impact":           np.abs(coefs),
        "coef_ci_2.5%":         coef_lo,
        "coef_ci_97.5%":        coef_hi,
        "confidence_stability": stability,
        "avg_in_1star":         mean_1,
        "avg_in_5star":         mean_5,
        "diff_(1star-5star)":   mean_1 - mean_5,
    }).sort_values("impact_coefficient")

    model_stats = {
        "cv_r2_mean":   cv_r2.mean(),
        "cv_r2_std":    cv_r2.std(),
        "cv_rmse_mean": cv_rmse.mean(),
        "cv_rmse_std":  cv_rmse.std(),
        "intercept":    intercept,
    }

    return df, impact_df, coefs, stability, model_stats