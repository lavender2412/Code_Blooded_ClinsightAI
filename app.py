import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Page config & custom CSS ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinsightAI Dashboard",
    layout="wide",
    page_icon="🏥",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    letter-spacing: -0.02em;
}
.stApp { background-color: #f6f7f9; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #0f172a !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #ffffff;
    padding: 8px 10px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
    font-size: 0.88rem;
    color: #475569;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #0f172a !important;
    color: #f8fafc !important;
}
hr { border-color: #e2e8f0; }
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #0f172a;
    margin-bottom: 0;
}
.main-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 2px;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme helpers ──────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font=dict(family="DM Sans, sans-serif", color="#334155"),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    title_font=dict(family="DM Serif Display, serif", size=17, color="#0f172a"),
    margin=dict(l=16, r=16, t=48, b=16),
    hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc", font_family="DM Sans"),
)

NEG_COLOR     = "#ef4444"
POS_COLOR     = "#10b981"
CLASS_PALETTE = {"Systemic": "#ef4444", "Recurring": "#f59e0b", "Isolated": "#10b981"}

def apply_layout(fig, **kwargs):
    fig.update_layout(**{**PLOTLY_LAYOUT, **kwargs})
    fig.update_xaxes(gridcolor="#e2e8f0", zeroline=False, showline=False)
    fig.update_yaxes(gridcolor="#e2e8f0", zeroline=False, showline=False)
    return fig

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🏥 ClinsightAI</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Healthcare Review Intelligence Dashboard</p>', unsafe_allow_html=True)

# ── Load CSVs ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    theme_df   = pd.read_csv("theme_level_outputs.csv")
    reviews_df = pd.read_csv("review_level_outputs.csv")
    roadmap_df = pd.read_csv("task4_action_roadmap.csv")
    theme_df = theme_df.rename(columns={
        "impact_coefficient_topic_part": "impact_coefficient",
        "abs_impact_topic_part":         "abs_impact",
    })
    return theme_df, reviews_df, roadmap_df

theme_df, reviews_df, roadmap_df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 Global Filters")
selected_theme         = st.sidebar.selectbox("Theme", ["All"] + theme_df["theme_label"].tolist())
min_rating, max_rating = st.sidebar.slider("Rating Range", 1, 5, (1, 5))

st.sidebar.divider()
st.sidebar.markdown("## ⚠️ Task 3 Controls")
prob_thr      = st.sidebar.slider("Topic probability threshold", 0.05, 0.80, 0.30, 0.05)
isolated_thr  = st.sidebar.slider("Isolated cutoff (freq <)",    0.01, 0.20, 0.05, 0.01)
systemic_thr  = st.sidebar.slider("Systemic cutoff (freq ≥)",    0.05, 0.60, 0.20, 0.05)
risk_mode     = st.sidebar.selectbox("Risk Score Mode", [
    "Simple (Frequency × |Impact|)", "Confidence-adjusted (× Confidence)"
])
show_only     = st.sidebar.selectbox("Show", ["All", "Systemic only", "Recurring only", "Isolated only"])
impact_filter = st.sidebar.selectbox("Impact Filter", ["All", "Negative impact only", "Positive impact only"])

# ── Global filter ─────────────────────────────────────────────────────────────
filtered = reviews_df.copy()
filtered = filtered[(filtered["Ratings"] >= min_rating) & (filtered["Ratings"] <= max_rating)]
if selected_theme != "All":
    filtered = filtered[filtered["theme_label"] == selected_theme]

# ── KPI bar ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Reviews",        f"{len(filtered):,}")
k2.metric("Avg Actual Rating",    f"{filtered['Ratings'].mean():.2f} ⭐")
k3.metric("Avg Predicted Rating", f"{filtered['predicted_rating'].mean():.2f} ⭐")
top_risk = theme_df.sort_values("risk_score", ascending=False).iloc[0]
k4.metric("Highest Risk Theme",   top_risk["theme_label"], delta=top_risk["issue_class"])

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
        fig = go.Figure(go.Bar(
            x=freq_data.values, y=freq_data.index, orientation="h",
            marker=dict(color=freq_data.values, colorscale="Blues",
                        showscale=False, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Frequency: %{x:.3f}<extra></extra>",
        ))
        apply_layout(fig, title="Theme Frequency (Dominant Topic)",
                     xaxis_title="Proportion of Reviews", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Pie(
            labels=theme_df["theme_label"],
            values=theme_df["dominant_topic_frequency"],
            hole=0.45,
            marker=dict(colors=px.colors.qualitative.Pastel),
            textfont=dict(size=12),
            hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
        ))
        apply_layout(fig, title="Theme Distribution", height=360,
                     legend=dict(orientation="v", x=1.02, y=0.5))
        fig.update_traces(textposition="inside")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Rating Impact
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Impact Coefficients — How Each Theme Affects Ratings")
    st.caption("Positive = lifts rating | Negative = hurts rating")

    impact_display = theme_df[[
        "theme_label", "impact_coefficient", "coef_ci_2.5%", "coef_ci_97.5%", "confidence_stability"
    ]].copy().sort_values("impact_coefficient")
    impact_display.columns = ["Theme", "Coefficient", "CI Low", "CI High", "Stability"]

    # ── Lollipop chart (safe column names for itertuples) ─────────────────────
    impact_plot = impact_display.copy()

    fig = go.Figure()

    for i, row in enumerate(impact_plot.itertuples(index=False)):
        color = POS_COLOR if row.Coefficient >= 0 else NEG_COLOR

        # CI range line (light gray, drawn behind)
        fig.add_shape(
            type="line",
            x0=row._2, x1=row._3,  # CI Low, CI High by position
            y0=i, y1=i,
            line=dict(color="#cbd5e1", width=2),
            layer="below",
        )
        # Stem from zero to coefficient value
        fig.add_shape(
            type="line",
            x0=0, x1=row.Coefficient,
            y0=i, y1=i,
            line=dict(color=color, width=2.5),
        )

    # Dots with coefficient labels
    fig.add_trace(go.Scatter(
        x=impact_plot["Coefficient"],
        y=list(range(len(impact_plot))),
        mode="markers+text",
        marker=dict(
            color=[POS_COLOR if c >= 0 else NEG_COLOR for c in impact_plot["Coefficient"]],
            size=16,
            line=dict(color="#ffffff", width=2.5),
        ),
    text=[f"{c:+.4f}" for c in impact_plot["Coefficient"]],
    textposition=[
        "top left" if c >= 0 else "top right"
        for c in impact_plot["Coefficient"]
    ],
    textfont=dict(size=11, color="#475569", family="DM Sans"),
        hovertemplate="<b>%{customdata}</b><br>Coefficient: %{x:.4f}<extra></extra>",
        customdata=impact_plot["Theme"],
        showlegend=False,
    ))

    # Legend proxies
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=POS_COLOR, size=12),
        name="↑ Lifts rating",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=NEG_COLOR, size=12),
        name="↓ Hurts rating",
    ))

    fig.add_vline(x=0, line_width=1.5, line_color="#64748b")

    apply_layout(fig,
        title="Theme Impact on Star Rating",
        xaxis_title="Regression Coefficient",
        height=420,
        margin=dict(l=320, r=80, t=48, b=48),  # ← wider left margin
        legend=dict(orientation="h", y=1.08, x=0),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(impact_plot))),
            ticktext=impact_plot["Theme"].tolist(),
            tickfont=dict(size=12),
            automargin=True,   # ← also add this
            gridcolor="#e2e8f0",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table below chart
    st.dataframe(impact_display.round(4), use_container_width=True)

    st.divider()
    st.subheader("1-Star vs 5-Star Theme Drivers")
    star_df = theme_df[["theme_label", "avg_in_1star", "avg_in_5star"]].melt(
        id_vars="theme_label", var_name="Star Group", value_name="Avg Topic Probability"
    )
    star_df["Star Group"] = star_df["Star Group"].map({
        "avg_in_1star": "1-Star Reviews", "avg_in_5star": "5-Star Reviews"
    })
    fig = px.bar(
        star_df, x="Avg Topic Probability", y="theme_label",
        color="Star Group",
        color_discrete_map={
            "1-Star Reviews": "#86efac",   # light green
            "5-Star Reviews": "#16a34a",   # deep green
        },
        barmode="group",
        orientation="h",
    )
    apply_layout(fig, title="Avg Topic Probability: 1-Star vs 5-Star Reviews",
                 xaxis_title="Avg Topic Probability", yaxis_title="",
                 height=380,
                 margin=dict(l=260, r=40, t=48, b=40),
                 legend=dict(title="", orientation="h", y=1.08))
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)
    
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

    CLASS_BG = {"Systemic": "#fee2e2", "Recurring": "#fef9c3", "Isolated": "#dcfce7"}
    def color_class(val):
        return f"background-color: {CLASS_BG.get(val, '')}; font-weight: 600;"

    st.dataframe(
        risk_display.style.applymap(color_class, subset=["issue_class"]),
        use_container_width=True,
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Score by Theme")
        rs = theme_df.sort_values("risk_score", ascending=False)
        fig = go.Figure(go.Bar(
            x=rs["risk_score"], y=rs["theme_label"], orientation="h",
            marker=dict(
                color=rs["risk_score"],
                colorscale=[[0, "#fca5a5"], [1, "#b91c1c"]],
                showscale=False,
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>Risk: %{x:.4f}<extra></extra>",
        ))
        apply_layout(fig, title="Risk = Frequency × |Impact| × Stability",
                     xaxis_title="Risk Score", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Severity Heatmap")
        sev = theme_df[["theme_label", "severity_score"]].set_index("theme_label")
        fig = go.Figure(go.Heatmap(
            z=sev["severity_score"].values.reshape(-1, 1),
            y=sev.index.tolist(),
            x=["Severity Score"],
            colorscale="Reds",
            text=[[f"{v:.4f}"] for v in sev["severity_score"].values],
            texttemplate="%{text}",
            showscale=True,
        ))
        apply_layout(fig, title="Severity Score", height=360)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Issue Classification Breakdown")
    class_counts = theme_df["issue_class"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    fig = px.bar(
        class_counts, x="Class", y="Count",
        color="Class", color_discrete_map=CLASS_PALETTE,
        text="Count",
    )
    fig.update_traces(textposition="outside")
    apply_layout(fig, title="Themes by Classification",
                 showlegend=False, xaxis_title="", height=320)
    st.plotly_chart(fig, use_container_width=True)


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
    abs_impacts        = [abs(impact_map.get(topic_id_to_theme.get(i, ""), 0.0)) for i in range(1, K + 1)]
    high_impact_cutoff = np.median(abs_impacts)

    def classify_live(freq, abs_impact):
        if freq < isolated_thr:
            return "Isolated"
        if freq >= systemic_thr and abs_impact >= high_impact_cutoff:
            return "Systemic"
        return "Recurring"

    rows = []
    for idx, topic_id in enumerate(range(1, K + 1)):
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

    if show_only == "Systemic only":
        task3_df = task3_df[task3_df["issue_class"] == "Systemic"]
    elif show_only == "Recurring only":
        task3_df = task3_df[task3_df["issue_class"] == "Recurring"]
    elif show_only == "Isolated only":
        task3_df = task3_df[task3_df["issue_class"] == "Isolated"]
    if impact_filter == "Negative impact only":
        task3_df = task3_df[task3_df["impact_coefficient"] < 0]
    elif impact_filter == "Positive impact only":
        task3_df = task3_df[task3_df["impact_coefficient"] > 0]
    task3_df = task3_df.sort_values(risk_col, ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Threshold",    f"{prob_thr:.2f}")
    c2.metric("Themes shown", f"{len(task3_df)}")
    c3.metric("Top Risk",     task3_df.iloc[0]["theme_label"] if len(task3_df) > 0 else "—")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Frequency by Theme")
        fig = px.bar(
            task3_df, x="present_frequency", y="theme_label",
            orientation="h", color_discrete_sequence=["#0ea5e9"],
        )
        apply_layout(fig, title="Topic Presence Frequency",
                     xaxis_title="Frequency", height=340, showlegend=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader(f"Risk Score ({risk_mode.split('(')[0].strip()})")
        fig = px.bar(
            task3_df, x=risk_col, y="theme_label",
            orientation="h", color="issue_class",
            color_discrete_map=CLASS_PALETTE,
        )
        apply_layout(fig, title="Risk Score by Theme", xaxis_title="Risk Score",
                     height=340, legend=dict(title="Class", orientation="h", y=1.1))
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.dataframe(task3_df[[
        "topic_id", "theme_label", "issue_class",
        "reviews_with_topic", "present_frequency",
        "impact_coefficient", "confidence", risk_col
    ]], use_container_width=True)

    st.divider()
    st.subheader("Drilldown: Reviews triggering a selected theme")
    theme_to_topic_id = {v: k for k, v in topic_id_to_theme.items()}
    theme_pick4 = st.selectbox(
        "Select theme",
        task3_df["theme_label"].tolist() if len(task3_df) > 0 else theme_df["theme_label"].tolist(),
        key="t4_theme",
    )
    picked_prob_col = f"topic_prob_{theme_to_topic_id.get(theme_pick4, 1)}"
    subset = reviews_df[reviews_df[picked_prob_col] > prob_thr].sort_values("Ratings")
    st.write(f"Reviews where **{theme_pick4}** prob > {prob_thr:.2f} — Count: {len(subset):,}")
    st.dataframe(
        subset[["Feedback", "Ratings", "predicted_rating", "residual",
                "theme_label", picked_prob_col]].head(150),
        use_container_width=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Action Roadmap (Task 4)
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Task 4 — Business Action Roadmap")

    st.subheader("Prioritized Improvement Roadmap (Theme Level)")
    show_bucket = st.selectbox("Filter by effort bucket", ["All", "Quick Wins", "High Effort"])
    view = roadmap_df.copy()
    if show_bucket != "All":
        view = view[view["bucket"] == show_bucket]
    st.dataframe(
        view[["priority_rank", "theme_label", "issue_class", "risk_score", "effort", "recommendation"]],
        use_container_width=True,
    )

    fig = px.scatter(
        roadmap_df, x="priority_rank", y="risk_score",
        size="risk_score", color="effort", text="theme_label",
        color_discrete_sequence=px.colors.qualitative.Set2,
        size_max=40,
    )
    fig.update_traces(textposition="top center", textfont_size=11)
    apply_layout(fig, title="Priority vs. Risk Score",
                 xaxis_title="Priority Rank", yaxis_title="Risk Score",
                 height=360, legend=dict(title="Effort"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("KPIs for Selected Theme")
    theme_pick5 = st.selectbox("Pick a theme to view KPIs", roadmap_df["theme_label"].tolist(), key="t5_kpi")
    kpi_row     = roadmap_df[roadmap_df["theme_label"] == theme_pick5].iloc[0]
    kpis        = json.loads(kpi_row["kpis"])
    st.markdown(f"**Recommendation:** {kpi_row['recommendation']}")
    st.markdown(f"**Effort:** `{kpi_row['effort']}` | **Risk score:** {kpi_row['risk_score']:.4f}")
    for item in kpis:
        st.write(f"- **{item['kpi']}** → Target: **{item['target']}**")

    st.divider()
    st.subheader("Per-Review Recommendations")
    t4_min, t4_max = st.slider("Filter rating range", 1, 5, (1, 5), key="t5_rating")
    theme_filter_t5 = st.selectbox(
        "Filter by review theme for action",
        ["All"] + sorted(reviews_df["review_theme_for_action"].dropna().unique().tolist()),
        key="t5_theme",
    )
    t5_filtered = reviews_df[
        (reviews_df["Ratings"] >= t4_min) & (reviews_df["Ratings"] <= t4_max)
    ].copy()
    if theme_filter_t5 != "All":
        t5_filtered = t5_filtered[t5_filtered["review_theme_for_action"] == theme_filter_t5]
    t5_filtered = t5_filtered.sort_values("kpi_urgency_score", ascending=False)

    st.caption("Highest urgency reviews appear first (low rating + high priority theme).")
    st.dataframe(t5_filtered[[
        "Feedback", "Ratings", "predicted_rating", "residual",
        "review_theme_for_action", "theme_priority_rank",
        "effort_estimate", "kpi_urgency_score"
    ]].head(200), use_container_width=True)

    st.divider()
    st.subheader("Drilldown: One Review → Recommendation + KPI Pack")
    if len(t5_filtered) > 0:
        t5_idx = st.number_input("Pick a row index", min_value=0,
                                 max_value=len(t5_filtered) - 1, value=0, key="t5_idx")
        t5_row = t5_filtered.iloc[int(t5_idx)]
        st.markdown(f"**Review:** {t5_row['Feedback']}")
        st.markdown(
            f"**Rating:** {t5_row['Ratings']} | "
            f"**Predicted:** {t5_row['predicted_rating']:.2f} | "
            f"**Residual:** {t5_row['residual']:.2f}"
        )
        st.markdown(
            f"**Theme:** {t5_row['review_theme_for_action']} | "
            f"**Effort:** {t5_row['effort_estimate']} | "
            f"**Urgency score:** {t5_row['kpi_urgency_score']:.2f}"
        )
        st.markdown("### Recommendation")
        st.write(t5_row["recommendation"])
        st.markdown("### KPIs to Track")
        kpis = json.loads(t5_row["kpi_pack"]) if isinstance(t5_row["kpi_pack"], str) else []
        if not kpis:
            st.info("No KPI pack found for this theme.")
        else:
            for item in kpis:
                st.write(f"- **{item['kpi']}** → Target: **{item['target']}**")
        try:
            q = json.loads(t5_row.get("quick_wins", "[]"))
            h = json.loads(t5_row.get("high_effort_fixes", "[]"))
            if q:
                st.markdown("### Quick Wins")
                for bullet in q:
                    st.write(f"- {bullet}")
            if h:
                st.markdown("### High-Effort Fixes")
                for bullet in h:
                    st.write(f"- {bullet}")
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

    rating_counts = filtered["Ratings"].value_counts().sort_index()
    fig = px.bar(
        x=rating_counts.index, y=rating_counts.values,
        labels={"x": "Star Rating", "y": "Count"},
        color=rating_counts.values,
        color_continuous_scale=["#ef4444", "#f97316", "#eab308", "#84cc16", "#10b981"],
        text=rating_counts.values,
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    apply_layout(fig, title="Rating Distribution (filtered)", height=280,
                 showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(filtered[[
        "Feedback", "Actual Rating", "Predicted Rating", "Residual",
        "theme", "dominant_contributor_theme"
    ]].head(300), use_container_width=True)

    st.divider()
    st.subheader("Drill Down: Single Review")
    if len(filtered) > 0:
        idx = st.number_input("Pick a row index", min_value=0,
                              max_value=len(filtered) - 1, value=0, key="t6_idx")
        row = filtered.iloc[int(idx)]
        st.markdown(f"**Review:** {row['Feedback']}")
        st.markdown(
            f"**Rating:** {row['Actual Rating']} | "
            f"**Predicted:** {row['Predicted Rating']:.2f} | "
            f"**Residual:** {row['Residual']:.2f}"
        )
        st.markdown(f"**Dominant Theme:** {row['theme']}")
        st.markdown(f"**Dominant Contributor:** {row['dominant_contributor_theme']}")

        prob_cols    = [c for c in filtered.columns if c.startswith("topic_prob_")]
        contrib_cols = [c for c in filtered.columns if c.startswith("contrib_")]
        c1, c2 = st.columns(2)

        with c1:
            st.write("Topic Probabilities")
            vals = row[prob_cols].rename(lambda x: x.replace("topic_prob_", "Topic "))
            fig = px.bar(x=vals.index, y=vals.values,
                         color_discrete_sequence=["#0ea5e9"])
            apply_layout(fig, title="", xaxis_title="", yaxis_title="Probability",
                         height=280, showlegend=False)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.write("Topic Contributions")
            cvals = row[contrib_cols].rename(lambda x: x.replace("contrib_", "Topic "))
            colors_c = [POS_COLOR if v >= 0 else NEG_COLOR for v in cvals.values]
            fig = go.Figure(go.Bar(
                x=cvals.index, y=cvals.values,
                marker=dict(color=colors_c, line=dict(width=0)),
            ))
            apply_layout(fig, title="", xaxis_title="", yaxis_title="Contribution",
                         height=280, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
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
            badge = {"Systemic": "🔴", "Recurring": "🟡", "Isolated": "🟢"}.get(
                theme["issue_class"], "⚪"
            )
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
            mime="application/json",
        )
    else:
        st.warning("clinsight_output.json not found. Run `python3 run_pipeline.py` first.")
