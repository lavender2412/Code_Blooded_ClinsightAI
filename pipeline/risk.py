import numpy as np
import pandas as pd

from pipeline.topic_model import N_TOPICS, TOPIC_LABELS
from pipeline.impact import PROB_THRESHOLD

def run_risk(topic_prob, coefs, stability):
    """
    Compute severity, risk score and systemic classification per theme.
    Returns risk_df — one row per theme.
    """
    present      = (topic_prob > PROB_THRESHOLD).astype(int)
    present_freq = present.mean(axis=0)

    severity = present_freq * np.abs(coefs)
    risk     = severity * stability

    def classify(freq, abs_impact):
        if freq < 0.05:
            return "Isolated"
        if freq >= 0.20 and abs_impact >= np.median(np.abs(coefs)):
            return "Systemic"
        return "Recurring"

    risk_df = pd.DataFrame({
        "topic_id":                      np.arange(1, N_TOPICS + 1),
        "theme_label":                   [TOPIC_LABELS[i] for i in range(1, N_TOPICS + 1)],
        "present_frequency_(prob>thr)":  present_freq,
        "abs_impact":                    np.abs(coefs),
        "severity_score":                severity,
        "risk_score":                    risk,
        "confidence_stability":          stability,
        "issue_class":                   [classify(f, a) for f, a in zip(present_freq, np.abs(coefs))]
    }).sort_values("risk_score", ascending=False).reset_index(drop=True)

    return risk_df