import json
import numpy as np
import pandas as pd

def generate_json_output(theme_level: pd.DataFrame, review_level: pd.DataFrame, topic_labels: dict) -> dict:
    """
    Generates the structured JSON output matching the problem statement format.
    Call this after running the pipeline, before saving CSVs.
    """

    # ── Clinic Summary ────────────────────────────────────────────────────────
    overall_rating_mean = round(float(review_level["Ratings"].mean()), 2)

    # Top 2 risk themes = primary operational risks
    top_risk_themes = (
        theme_level.sort_values("risk_score", ascending=False)
        ["theme_label"].head(2).tolist()
    )

    clinic_summary = {
        "overall_rating_mean":  overall_rating_mean,
        "total_reviews":        int(len(review_level)),
        "avg_predicted_rating": round(float(review_level["predicted_rating"].mean()), 2),
        "primary_risk_themes":  top_risk_themes
    }

    # ── Theme Analysis ────────────────────────────────────────────────────────
    theme_analysis = []

    for _, row in theme_level.iterrows():
        theme_label = row["theme_label"]

        # Pull up to 3 real review samples for this theme
        # Use reviews where this theme is dominant AND rating <= 2 (negative signal)
        theme_reviews = review_level[
            (review_level["theme_label"] == theme_label) &
            (review_level["Ratings"] <= 2)
        ]["Feedback"].dropna().head(3).tolist()

        # Fallback to any review for this theme if no negative ones
        if not theme_reviews:
            theme_reviews = review_level[
                review_level["theme_label"] == theme_label
            ]["Feedback"].dropna().head(3).tolist()

        # Truncate long reviews to 80 chars for readability
        evidence_samples = [str(r)[:80] + "..." if len(str(r)) > 80 else str(r) for r in theme_reviews]

        theme_analysis.append({
            "theme":                theme_label,
            "frequency_percentage": round(float(row["dominant_topic_frequency"]) * 100, 1),
            "present_frequency":    round(float(row["present_frequency_(prob>thr)"]) * 100, 1),
            "rating_impact":        round(float(row["impact_coefficient"]), 3),
            "severity_score":       round(float(row["severity_score"]), 4),
            "risk_score":           round(float(row["risk_score"]), 4),
            "issue_class":          row["issue_class"],
            "confidence_stability": round(float(row["confidence_stability"]), 3),
            "evidence_samples":     evidence_samples
        })

    # ── Improvement Roadmap ───────────────────────────────────────────────────
    # Map themes to actionable recommendations
    RECOMMENDATIONS = {
        "Wait Time & Operational Efficiency": {
            "recommendation":      "Optimise appointment scheduling workflow and reduce patient wait times",
            "expected_rating_lift": "+0.6",
        },
        "Emergency Service Failures": {
            "recommendation":      "Establish emergency response protocols and dedicated triage staff",
            "expected_rating_lift": "+0.5",
        },
        "Clinical Care & Treatment Quality": {
            "recommendation":      "Implement peer review program and clinical quality audits",
            "expected_rating_lift": "+0.4",
        },
        "Overall Service & Facility Cleanliness": {
            "recommendation":      "Introduce hourly facility checks and staff service training",
            "expected_rating_lift": "+0.3",
        },
        "Consultation & Positive Patient Experience": {
            "recommendation":      "Expand consultation time and patient communication training",
            "expected_rating_lift": "+0.2",
        },
    }

    improvement_roadmap = []
    # Sort by risk score — highest risk = highest priority
    prioritised = theme_level[theme_level["impact_coefficient"] < 0].sort_values(
        "risk_score", ascending=False
    )

    for priority, (_, row) in enumerate(prioritised.iterrows(), start=1):
        label = row["theme_label"]
        rec   = RECOMMENDATIONS.get(label, {
            "recommendation":      f"Address recurring complaints related to {label}",
            "expected_rating_lift": "+0.3",
        })
        improvement_roadmap.append({
            "priority":             priority,
            "theme":                label,
            "recommendation":       rec["recommendation"],
            "expected_rating_lift": rec["expected_rating_lift"],
            "confidence":           round(float(row["confidence_stability"]), 3),
            "issue_class":          row["issue_class"],
        })

    # ── Assemble final output ─────────────────────────────────────────────────
    output = {
        "clinic_summary":      clinic_summary,
        "theme_analysis":      theme_analysis,
        "improvement_roadmap": improvement_roadmap,
    }

    return output


def save_json(output: dict, path: str = "clinsight_output.json"):
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Saved: {path}")