import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

st.set_page_config(page_title="ClinsightAI Dashboard", layout="wide")
st.title("🏥 ClinsightAI — Healthcare Review Intelligence Dashboard")

# ── Load CSVs ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    theme_df   = pd.read_csv("theme_level_outputs.csv")
    reviews_df = pd.read_csv("review_level_outputs.csv")
    roadmap_df = pd.read_csv("task4_action_roadmap.csv")
    
    # Rename columns that changed when TF-IDF was added
    theme_df = theme_df.rename(columns={
        "impact_coefficient_topic_part": "impact_coefficient",
        "abs_impact_topic_part":         "abs_impact",
    })
    
    return theme_df, reviews_df, roadmap_df


theme_df, reviews_df, roadmap_df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Global Filters")
selected_theme      = st.sidebar.selectbox("Select a Theme", ["All"] + theme_df["theme_label"].tolist())
min_rating, max_rating = st.sidebar.slider("Rating Range", 1, 5, (1, 5))

st.sidebar.divider()
st.sidebar.subheader("⚠️ Task 3 Controls")
prob_thr      = st.sidebar.slider("Topic probability threshold", 0.05, 0.80, 0.30, 0.05)
isolated_thr  = st.sidebar.slider("Isolated cutoff (freq <)",    0.01, 0.20, 0.05, 0.01)
systemic_thr  = st.sidebar.slider("Systemic cutoff (freq ≥)",    0.05, 0.60, 0.20, 0.05)
risk_mode     = st.sidebar.selectbox("Risk Score Mode", [
    "Simple (Frequency × |Impact|)", "Confidence-adjusted (× Confidence)"
])
show_only     = st.sidebar.selectbox("Show", ["All", "Systemic only", "Recurring only", "Isolated only"])
impact_filter = st.sidebar.selectbox("Impact Filter", ["All", "Negative impact only", "Positive impact only"])

# ── Global filtered reviews ───────────────────────────────────────────────────
filtered = reviews_df.copy()
filtered = filtered[(filtered["Ratings"] >= min_rating) & (filtered["Ratings"] <= max_rating)]
if selected_theme != "All":
    filtered = filtered[filtered["theme_label"] == selected_theme]

# ── KPI bar ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Reviews (filtered)",  f"{len(filtered):,}")
k2.metric("Avg Actual Rating",         f"{filtered['Ratings'].mean():.2f} ⭐")
k3.metric("Avg Predicted Rating",      f"{filtered['predicted_rating'].mean():.2f} ⭐")
top_risk = theme_df.sort_values("risk_score", ascending=False).iloc[0]
k4.metric("Highest Risk Theme",        top_risk["theme_label"], delta=top_risk["issue_class"])

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Operational Themes",
    "📈 Rating Impact",
    "⚠️ Risk & Systemic Issues",
    "🔁 Recurring (Interactive)",
    "🗺️ Action Roadmap",
    "🔬 Review Explorer",
    "🧾 JSON Output",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Theme Discovery
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Theme Frequency Distribution")
    col1, col2 = st.columns(2)

    with col1:
        freq_data = theme_df.set_index("theme_label")["dominant_topic_frequency"].sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=freq_data.values, y=freq_data.index, palette="Blues_d", ax=ax)
        ax.set_xlabel("Proportion of Reviews")
        ax.set_title("Theme Frequency (Dominant Topic)")
        st.pyplot(fig); plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(theme_df["dominant_topic_frequency"], labels=theme_df["theme_label"],
               autopct='%1.1f%%', startangle=140)
        ax.set_title("Theme Distribution")
        st.pyplot(fig); plt.close(fig)
# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Rating Impact
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Impact Coefficients — How Each Theme Affects Ratings")
    st.caption("Positive = lifts rating | Negative = hurts rating")

    impact_display = theme_df[[
        "theme_label", "impact_coefficient", "coef_ci_2.5%", "coef_ci_97.5%", "confidence_stability"
    ]].copy().sort_values("impact_coefficient")
    impact_display.columns = ["Theme", "Coefficient", "CI Low (2.5%)", "CI High (97.5%)", "Stability"]

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#d73027" if c < 0 else "#1a9850" for c in impact_display["Coefficient"]]
        ax.barh(impact_display["Theme"], impact_display["Coefficient"], color=colors)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel("Regression Coefficient")
        ax.set_title("Theme Impact on Star Rating")
        ax.legend(handles=[
            mpatches.Patch(color='#d73027', label='Hurts rating'),
            mpatches.Patch(color='#1a9850', label='Lifts rating')
        ])
        st.pyplot(fig); plt.close(fig)
    with col2:
        st.dataframe(impact_display.round(4), use_container_width=True)

    st.divider()
    st.subheader("1-Star vs 5-Star Theme Drivers")
    star_df = theme_df[["theme_label", "avg_in_1star", "avg_in_5star"]].melt(
        id_vars="theme_label", var_name="Star Group", value_name="Avg Topic Probability"
    )
    star_df["Star Group"] = star_df["Star Group"].map({
        "avg_in_1star": "1-Star Reviews", "avg_in_5star": "5-Star Reviews"
    })
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=star_df, x="theme_label", y="Avg Topic Probability",
                hue="Star Group", palette=["#d73027", "#1a9850"], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)
    ax.set_title("Avg Topic Probability: 1-Star vs 5-Star Reviews")
    st.pyplot(fig); plt.close(fig)

    st.divider()
    
    # DEBUG — remove after fixing
    st.markdown("### Debug Info")
    st.write("CV R2 mean value:", theme_df['cv_r2_mean'].iloc[0])
    st.write("CV RMSE mean value:", theme_df['cv_rmse_mean'].iloc[0])
    st.write("CSV columns:", theme_df.columns.tolist())
    st.dataframe(theme_df[['theme_label', 'cv_r2_mean', 'cv_rmse_mean']].head())

    st.subheader("Model Robustness")
    m1, m2, m3 = st.columns(3)
    m1.metric("Cross-validated R²",   f"{theme_df['cv_r2_mean'].iloc[0]:.3f}")
    m2.metric("Cross-validated RMSE", f"{theme_df['cv_rmse_mean'].iloc[0]:.3f}")
    m3.metric("Intercept",            f"{theme_df['intercept'].iloc[0]:.3f}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Risk & Systemic Issues (static, from pipeline)
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Theme Risk Ranking")
    risk_display = theme_df[[
        "topic_id", "theme_label", "issue_class",
        "present_frequency_(prob>thr)", "impact_coefficient",
        "severity_score", "risk_score", "confidence_stability"
    ]].copy()

    def color_class(val):
        return {"Systemic":  "background-color: #f8d7da",
                "Recurring": "background-color: #fff3cd",
                "Isolated":  "background-color: #d4edda"}.get(val, "")

    st.dataframe(risk_display.style.applymap(color_class, subset=["issue_class"]),
                 use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Score by Theme")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=theme_df.sort_values("risk_score", ascending=False),
                    x="risk_score", y="theme_label", palette="Reds_d", ax=ax)
        ax.set_title("Risk = Frequency × |Impact| × Stability")
        st.pyplot(fig); plt.close(fig)
    with col2:
        st.subheader("Severity Heatmap")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(theme_df.set_index("theme_label")[["severity_score"]],
                    annot=True, fmt=".4f", cmap="Reds", ax=ax)
        st.pyplot(fig); plt.close(fig)

    st.divider()
    st.subheader("Issue Classification Breakdown")
    class_counts = theme_df["issue_class"].value_counts()
    palette_map  = {"Systemic": "#d73027", "Recurring": "#fc8d59", "Isolated": "#1a9850"}
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(x=class_counts.index, y=class_counts.values,
                palette=[palette_map.get(c, "gray") for c in class_counts.index], ax=ax)
    ax.set_ylabel("Number of Themes")
    ax.set_title("Themes by Classification")
    st.pyplot(fig); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Recurring Issues (Interactive)
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Recurring & Systemic Issues — Interactive")
    st.caption("Adjust thresholds in the sidebar to recompute classifications live.")

    topic_prob_cols   = [c for c in reviews_df.columns if c.startswith("topic_prob_")]
    K                 = len(topic_prob_cols)
    impact_map        = dict(zip(theme_df["theme_label"], theme_df["impact_coefficient"]))
    conf_map          = dict(zip(theme_df["theme_label"], theme_df["confidence_stability"]))
    topic_id_to_theme = dict(zip(theme_df["topic_id"], theme_df["theme_label"]))

    present_mat        = (reviews_df[topic_prob_cols].values > prob_thr).astype(int)
    present_freq_live  = present_mat.mean(axis=0)
    reviews_with_topic = present_mat.sum(axis=0)
    abs_impacts        = [abs(impact_map.get(topic_id_to_theme.get(i, ""), 0.0)) for i in range(1, K+1)]
    high_impact_cutoff = np.median(abs_impacts)

    def classify_live(freq, abs_impact):
        if freq < isolated_thr:                                    return "Isolated"
        if freq >= systemic_thr and abs_impact >= high_impact_cutoff: return "Systemic"
        return "Recurring"

    rows = []
    for idx, topic_id in enumerate(range(1, K+1)):
        theme   = topic_id_to_theme.get(topic_id, f"Topic {topic_id}")
        imp     = impact_map.get(theme, 0.0)
        abs_imp = abs(imp)
        freq    = float(present_freq_live[idx])
        conf    = float(conf_map.get(theme, 1.0))
        rows.append({
            "topic_id":              topic_id,
            "theme_label":           theme,
            "issue_class":           classify_live(freq, abs_imp),
            "reviews_with_topic":    int(reviews_with_topic[idx]),
            "present_frequency":     freq,
            "impact_coefficient":    imp,
            "abs_impact":            abs_imp,
            "confidence":            conf,
            "risk_score_simple":     freq * abs_imp,
            "risk_score_confidence": freq * abs_imp * conf,
        })

    task3_df = pd.DataFrame(rows)
    risk_col = "risk_score_simple" if risk_mode.startswith("Simple") else "risk_score_confidence"

    if show_only == "Systemic only":    task3_df = task3_df[task3_df["issue_class"] == "Systemic"]
    elif show_only == "Recurring only": task3_df = task3_df[task3_df["issue_class"] == "Recurring"]
    elif show_only == "Isolated only":  task3_df = task3_df[task3_df["issue_class"] == "Isolated"]
    if impact_filter == "Negative impact only": task3_df = task3_df[task3_df["impact_coefficient"] < 0]
    elif impact_filter == "Positive impact only": task3_df = task3_df[task3_df["impact_coefficient"] > 0]
    task3_df = task3_df.sort_values(risk_col, ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Threshold",    f"{prob_thr:.2f}")
    c2.metric("Themes shown", f"{len(task3_df)}")
    c3.metric("Top Risk",     task3_df.iloc[0]["theme_label"] if len(task3_df) > 0 else "—")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Frequency by Theme")
        st.bar_chart(task3_df.set_index("theme_label")["present_frequency"])
    with right:
        st.subheader(f"Risk Score ({risk_mode})")
        st.bar_chart(task3_df.set_index("theme_label")[risk_col])

    st.divider()
    st.dataframe(task3_df[[
        "topic_id", "theme_label", "issue_class",
        "reviews_with_topic", "present_frequency",
        "impact_coefficient", "confidence", risk_col
    ]], use_container_width=True)

    st.divider()
    st.subheader("Drilldown: Reviews triggering a selected theme")
    theme_to_topic_id = {v: k for k, v in topic_id_to_theme.items()}
    theme_pick4       = st.selectbox("Select theme", task3_df["theme_label"].tolist() if len(task3_df) > 0 else theme_df["theme_label"].tolist(), key="t4_theme")
    picked_prob_col   = f"topic_prob_{theme_to_topic_id.get(theme_pick4, 1)}"
    subset            = reviews_df[reviews_df[picked_prob_col] > prob_thr].sort_values("Ratings")
    st.write(f"Reviews where **{theme_pick4}** prob > {prob_thr:.2f} — Count: {len(subset):,}")
    st.dataframe(subset[["Feedback", "Ratings", "predicted_rating", "residual", "theme_label", picked_prob_col]].head(150),
                 use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Action Roadmap (Task 4)
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Task 4 — Business Action Roadmap")

    # Roadmap table
    st.subheader("Prioritized Improvement Roadmap (Theme Level)")
    show_bucket = st.selectbox("Filter by effort bucket", ["All", "Quick Wins", "High Effort"])
    view = roadmap_df.copy()
    if show_bucket != "All":
        view = view[view["bucket"] == show_bucket]
    st.dataframe(
        view[["priority_rank", "theme_label", "issue_class", "risk_score", "effort", "recommendation"]],
        use_container_width=True
    )

    # KPIs for selected theme
    st.subheader("KPIs for Selected Theme")
    theme_pick5 = st.selectbox("Pick a theme to view KPIs", roadmap_df["theme_label"].tolist(), key="t5_kpi")
    kpi_row     = roadmap_df[roadmap_df["theme_label"] == theme_pick5].iloc[0]
    kpis        = json.loads(kpi_row["kpis"])
    st.markdown(f"**Recommendation:** {kpi_row['recommendation']}")
    st.markdown(f"**Effort:** `{kpi_row['effort']}` | **Risk score:** {kpi_row['risk_score']:.4f}")
    for item in kpis:
        st.write(f"- **{item['kpi']}** → Target: **{item['target']}**")

    st.divider()

    # Per-review recommendations
    st.subheader("Per-Review Recommendations (Data Point Level)")
    t4_min, t4_max = st.slider("Filter rating range", 1, 5, (1, 5), key="t5_rating")
    theme_filter_t5 = st.selectbox(
        "Filter by review theme for action",
        ["All"] + sorted(reviews_df["review_theme_for_action"].dropna().unique().tolist()),
        key="t5_theme"
    )
    t5_filtered = reviews_df[(reviews_df["Ratings"] >= t4_min) & (reviews_df["Ratings"] <= t4_max)].copy()
    if theme_filter_t5 != "All":
        t5_filtered = t5_filtered[t5_filtered["review_theme_for_action"] == theme_filter_t5]
    t5_filtered = t5_filtered.sort_values("kpi_urgency_score", ascending=False)

    st.write("Tip: Highest urgency reviews appear first (low rating + high priority theme).")
    st.dataframe(t5_filtered[[
        "Feedback", "Ratings", "predicted_rating", "residual",
        "review_theme_for_action", "theme_priority_rank",
        "effort_estimate", "kpi_urgency_score"
    ]].head(200), use_container_width=True)

    st.divider()

    # Drilldown
    st.subheader("Drilldown: One Review → Recommendation + KPI Pack")
    if len(t5_filtered) > 0:
        t5_idx = st.number_input("Pick a row index (0..N-1 of filtered list)",
                                 min_value=0, max_value=len(t5_filtered)-1, value=0, key="t5_idx")
        t5_row = t5_filtered.iloc[int(t5_idx)]
        st.markdown(f"**Review:** {t5_row['Feedback']}")
        st.markdown(f"**Rating:** {t5_row['Ratings']} | **Predicted:** {t5_row['predicted_rating']:.2f} | **Residual:** {t5_row['residual']:.2f}")
        st.markdown(f"**Theme:** {t5_row['review_theme_for_action']} | **Effort:** {t5_row['effort_estimate']} | **Urgency score:** {t5_row['kpi_urgency_score']:.2f}")
        st.markdown("### Recommendation")
        st.write(t5_row["recommendation"])
        st.markdown("### KPIs to Track")
        kpis = json.loads(t5_row["kpi_pack"]) if isinstance(t5_row["kpi_pack"], str) else []
        if not kpis:
            st.info("No KPI pack found for this theme.")
        else:
            for item in kpis:
                st.write(f"- **{item['kpi']}** → Target: **{item['target']}**")
        # GPT quick wins / high effort (only populated if USE_GPT=True)
        try:
            q = json.loads(t5_row.get("quick_wins", "[]"))
            h = json.loads(t5_row.get("high_effort_fixes", "[]"))
            if q:
                st.markdown("### Quick Wins")
                for bullet in q: st.write(f"- {bullet}")
            if h:
                st.markdown("### High-Effort Fixes")
                for bullet in h: st.write(f"- {bullet}")
        except Exception:
            pass
    else:
        st.info("No reviews match the selected filters.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Review Explorer
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Review Explorer")
    st.caption("Filtered by sidebar. Sort by residual to find where the model under/over-predicts.")
    st.dataframe(filtered[[
        "Feedback", "Ratings", "predicted_rating", "residual",
        "theme_label", "dominant_contributor_theme",
        "most_positive_theme", "most_negative_theme"
    ]].head(300), use_container_width=True)

    st.divider()
    st.subheader("Drill Down: Single Review")
    if len(filtered) > 0:
        idx = st.number_input("Pick a row index", min_value=0, max_value=len(filtered)-1, value=0, key="t6_idx")
        row = filtered.iloc[int(idx)]
        st.markdown(f"**Review:** {row['Feedback']}")
        st.markdown(f"**Rating:** {row['Ratings']} | **Predicted:** {row['predicted_rating']:.2f} | **Residual:** {row['residual']:.2f}")
        st.markdown(f"**Dominant Theme:** {row['theme_label']}")
        st.markdown(f"**Dominant Contributor:** {row['dominant_contributor_theme']}")
        prob_cols    = [c for c in filtered.columns if c.startswith("topic_prob_")]
        contrib_cols = [c for c in filtered.columns if c.startswith("contrib_")]
        c1, c2 = st.columns(2)
        with c1:
            st.write("Topic Probabilities")
            st.bar_chart(row[prob_cols].rename(lambda x: x.replace("topic_prob_", "Topic ")))
        with c2:
            st.write("Topic Contributions")
            st.bar_chart(row[contrib_cols].rename(lambda x: x.replace("contrib_", "Topic ")))
    else:
        st.info("No reviews match the current filters.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — JSON Output
# ════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("🧾 Structured JSON Output")
    st.caption("Machine-readable output matching the problem statement format.")

    if os.path.exists("clinsight_output.json"):
        with open("clinsight_output.json") as f:
            data = json.load(f)

        st.markdown("### 🏥 Clinic Summary")
        cs = data["clinic_summary"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Rating Mean",  cs["overall_rating_mean"])
        c2.metric("Total Reviews",        cs["total_reviews"])
        c3.metric("Avg Predicted Rating", cs["avg_predicted_rating"])
        st.markdown("**Primary Risk Themes:**")
        for t in cs["primary_risk_themes"]:
            st.markdown(f"- 🔴 {t}")

        st.divider()
        st.markdown("### 📊 Theme Analysis")
        for theme in data["theme_analysis"]:
            badge = {"Systemic": "🔴", "Recurring": "🟡", "Isolated": "🟢"}.get(theme["issue_class"], "⚪")
            with st.expander(f"{badge} {theme['theme']} — Risk: {theme['risk_score']:.4f}"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Frequency",     f"{theme['frequency_percentage']}%")
                col2.metric("Rating Impact", theme["rating_impact"])
                col3.metric("Severity",      theme["severity_score"])
                col4.metric("Class",         theme["issue_class"])
                st.markdown("**Evidence samples:**")
                for s in theme["evidence_samples"]:
                    st.markdown(f"> {s}")

        st.divider()
        st.markdown("### 🗺️ Improvement Roadmap")
        for item in data["improvement_roadmap"]:
            with st.expander(f"Priority {item['priority']} — {item['theme']}"):
                st.markdown(f"**Recommendation:** {item['recommendation']}")
                st.markdown(f"**Effort:** `{item['effort']}`")
                col1, col2 = st.columns(2)
                col1.metric("Expected Rating Lift", item["expected_rating_lift"])
                col2.metric("Confidence",           item["confidence"])
                if item.get("kpis"):
                    st.markdown("**KPIs:**")
                    for kpi in item["kpis"]:
                        st.write(f"- **{kpi['kpi']}** → {kpi['target']}")

        st.divider()
        st.markdown("### Raw JSON")
        st.json(data)
        st.download_button(
            label="⬇️ Download clinsight_output.json",
            data=json.dumps(data, indent=2),
            file_name="clinsight_output.json",
            mime="application/json"
        )
    else:
        st.warning("clinsight_output.json not found. Run `python3 run_pipeline.py` first.")