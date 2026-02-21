import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx

st.set_page_config(page_title="ClinsightAI Dashboard", layout="wide")
st.title("🏥 ClinsightAI — Healthcare Review Intelligence Dashboard")

# ── Load pre-generated CSVs ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    theme_df   = pd.read_csv("theme_level_outputs.csv")
    reviews_df = pd.read_csv("review_level_outputs.csv")
    return theme_df, reviews_df

theme_df, reviews_df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")
selected_theme = st.sidebar.selectbox(
    "Select a Theme",
    ["All"] + theme_df["theme_label"].tolist()
)
min_rating, max_rating = st.sidebar.slider("Rating Range", 1, 5, (1, 5))

filtered = reviews_df.copy()
filtered = filtered[
    (filtered["Ratings"] >= min_rating) &
    (filtered["Ratings"] <= max_rating)
]
if selected_theme != "All":
    filtered = filtered[filtered["theme_label"] == selected_theme]

# ── Top KPI metrics ───────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Reviews (filtered)",  f"{len(filtered):,}")
k2.metric("Avg Actual Rating",         f"{filtered['Ratings'].mean():.2f} ⭐")
k3.metric("Avg Predicted Rating",      f"{filtered['predicted_rating'].mean():.2f} ⭐")
top_risk = theme_df.sort_values("risk_score", ascending=False).iloc[0]
k4.metric("Highest Risk Theme",        top_risk["theme_label"], delta=top_risk["issue_class"])

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Theme Discovery",
    "📈 Rating Impact",
    "⚠️ Risk & Systemic Issues",
    "🔬 Review Explorer"
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
        ax.pie(
            theme_df["dominant_topic_frequency"],
            labels=theme_df["theme_label"],
            autopct='%1.1f%%', startangle=140
        )
        ax.set_title("Theme Distribution")
        st.pyplot(fig); plt.close(fig)

    st.divider()
    st.subheader("Top Words per Topic")
    cols = st.columns(len(theme_df))
    for i, row in theme_df.reset_index().iterrows():
        with cols[i]:
            st.markdown(f"**{row['theme_label']}**")
            for w in row['top_words'].split(' | ')[:10]:
                st.markdown(f"- {w}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Rating Impact
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Impact Coefficients — How Each Theme Affects Ratings")
    st.caption("Positive = lifts rating | Negative = hurts rating")

    impact_display = theme_df[[
        "theme_label", "impact_coefficient",
        "coef_ci_2.5%", "coef_ci_97.5%", "confidence_stability"
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
        "avg_in_1star": "1-Star Reviews",
        "avg_in_5star": "5-Star Reviews"
    })
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=star_df, x="theme_label", y="Avg Topic Probability",
                hue="Star Group", palette=["#d73027", "#1a9850"], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)
    ax.set_title("Avg Topic Probability: 1-Star vs 5-Star Reviews")
    st.pyplot(fig); plt.close(fig)

    st.divider()
    st.subheader("Model Robustness")
    m1, m2, m3 = st.columns(3)
    m1.metric("Cross-validated R²",   f"{theme_df['cv_r2_mean'].iloc[0]:.3f}")
    m2.metric("Cross-validated RMSE", f"{theme_df['cv_rmse_mean'].iloc[0]:.3f}")
    m3.metric("Intercept",            f"{theme_df['intercept'].iloc[0]:.3f}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Risk & Systemic Issues
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Theme Risk Ranking")

    risk_display = theme_df[[
        "topic_id", "theme_label", "issue_class",
        "present_frequency_(prob>thr)", "impact_coefficient",
        "severity_score", "risk_score", "confidence_stability"
    ]].copy()

    def color_class(val):
        return {
            "Systemic":  "background-color: #f8d7da",
            "Recurring": "background-color: #fff3cd",
            "Isolated":  "background-color: #d4edda"
        }.get(val, "")

    st.dataframe(
        risk_display.style.applymap(color_class, subset=["issue_class"]),
        use_container_width=True
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score by Theme")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            data=theme_df.sort_values("risk_score", ascending=False),
            x="risk_score", y="theme_label", palette="Reds_d", ax=ax
        )
        ax.set_title("Risk = Frequency × |Impact| × Stability")
        st.pyplot(fig); plt.close(fig)

    with col2:
        st.subheader("Severity Heatmap")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            theme_df.set_index("theme_label")[["severity_score"]],
            annot=True, fmt=".4f", cmap="Reds", ax=ax
        )
        st.pyplot(fig); plt.close(fig)

    st.divider()
    st.subheader("Issue Classification Breakdown")
    class_counts = theme_df["issue_class"].value_counts()
    palette_map  = {"Systemic": "#d73027", "Recurring": "#fc8d59", "Isolated": "#1a9850"}
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(
        x=class_counts.index, y=class_counts.values,
        palette=[palette_map.get(c, "gray") for c in class_counts.index], ax=ax
    )
    ax.set_ylabel("Number of Themes")
    ax.set_title("Themes by Classification")
    st.pyplot(fig); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Review Explorer
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Review Explorer")
    st.caption("Filtered by sidebar. Sort by residual to find where the model under/over-predicts.")

    st.dataframe(
        filtered[[
            "Feedback", "Ratings", "predicted_rating", "residual",
            "theme_label", "dominant_contributor_theme",
            "most_positive_theme", "most_negative_theme"
        ]].head(300),
        use_container_width=True
    )

    st.divider()
    st.subheader("Drill Down: Single Review")

    if len(filtered) > 0:
        idx = st.number_input(
            "Pick a row index (0 to N-1 of filtered table)",
            min_value=0, max_value=len(filtered) - 1, value=0
        )
        row = filtered.iloc[int(idx)]

        st.markdown(f"**Review:** {row['Feedback']}")
        st.markdown(f"**Rating:** {row['Ratings']} | **Predicted:** {row['predicted_rating']:.2f} | **Residual:** {row['residual']:.2f}")
        st.markdown(f"**Dominant Theme:** {row['theme_label']}")
        st.markdown(f"**Dominant Contributor:** {row['dominant_contributor_theme']}")

        prob_cols    = [c for c in filtered.columns if c.startswith("topic_prob_")]
        contrib_cols = [c for c in filtered.columns if c.startswith("contrib_")]
        prob_series    = row[prob_cols].rename(lambda x: x.replace("topic_prob_", "Topic "))
        contrib_series = row[contrib_cols].rename(lambda x: x.replace("contrib_", "Topic "))

        c1, c2 = st.columns(2)
        with c1:
            st.write("Topic Probabilities")
            st.bar_chart(prob_series)
        with c2:
            st.write("Topic Contributions (coef × prob)")
            st.bar_chart(contrib_series)
    else:
        st.info("No reviews match the current filters.")