import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

N_TOPICS  = 5
TOP_WORDS = 12

TOPIC_LABELS = {
    1: "Clinical Care & Treatment Quality",
    2: "Overall Service & Facility Cleanliness",
    3: "Emergency Service Failures",
    4: "Consultation & Positive Patient Experience",
    5: "Wait Time & Operational Efficiency",
}

def run_lda(df, n_topics=N_TOPICS):
    """
    Vectorize clean_text, fit LDA, return enriched df + all topic artefacts.
    Returns:
        df                 — with dominant_topic, theme_label columns added
        topic_prob         — (n_reviews, n_topics) probability matrix
        topic_words        — dict {theme_label: [word, ...]}
        theme_counts       — value_counts of dominant theme
        theme_freq         — theme_counts / total
        topic_similarity   — (n_topics, n_topics) cosine sim matrix
    """
    vectorizer = CountVectorizer(max_df=0.90, min_df=5, stop_words="english")
    dtm = vectorizer.fit_transform(df["clean_text"])

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()
    topic_words = {}
    for idx, topic in enumerate(lda.components_):
        top_w = [feature_names[i] for i in topic.argsort()[:-TOP_WORDS - 1:-1]]
        topic_words[TOPIC_LABELS[idx + 1]] = top_w

    topic_prob = lda.transform(dtm)

    for k in range(n_topics):
        df[f"topic_prob_{k+1}"] = topic_prob[:, k]

    df["dominant_topic"] = topic_prob.argmax(axis=1) + 1   # 1-indexed
    df["theme_label"]    = df["dominant_topic"].map(TOPIC_LABELS)

    theme_counts = df["theme_label"].value_counts()
    theme_freq   = theme_counts / len(df)

    topic_similarity = cosine_similarity(lda.components_)

    return df, topic_prob, topic_words, theme_counts, theme_freq, topic_similarity