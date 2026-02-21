import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

N_TOPICS = 5

TOPIC_LABELS = {
    0: "Clinical Care Quality",
    1: "Overall Service & Cleanliness",
    2: "Emergency Service Issues",
    3: "Consultation Experience",
    4: "Wait Time & Efficiency"
}

def run_lda(df, n_topics=N_TOPICS):
    # Vectorize
    vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
    dtm = vectorizer.fit_transform(df['clean_text'])

    # LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=18, learning_method='batch')
    lda.fit(dtm)

    # Topic probabilities per review
    topic_probabilities = lda.transform(dtm)
    df['dominant_topic'] = topic_probabilities.argmax(axis=1)
    df['theme'] = df['dominant_topic'].map(TOPIC_LABELS)

    # Top words per topic
    feature_names = vectorizer.get_feature_names_out()
    topic_words = {}
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topic_words[TOPIC_LABELS[idx]] = top_words

    # Theme frequency
    theme_counts = df['theme'].value_counts()
    theme_freq   = theme_counts / len(df)

    # Severity (Task 1 version — no ratings needed)
    avg_prob       = topic_probabilities.mean(axis=0)
    ordered_themes = [TOPIC_LABELS[i] for i in range(n_topics)]
    ordered_freq   = theme_freq.reindex(ordered_themes).fillna(0).values
    severity_scores = ordered_freq * np.abs(avg_prob)

    import pandas as pd
    severity_df = pd.DataFrame({
        "Theme":           ordered_themes,
        "Frequency (%)":   (ordered_freq * 100).round(1),
        "Avg Probability": avg_prob.round(3),
        "Severity Score":  severity_scores.round(4)
    }).sort_values("Severity Score", ascending=False).reset_index(drop=True)

    # Topic similarity matrix (for network graph)
    topic_similarity = cosine_similarity(lda.components_)

    return df, topic_probabilities, topic_words, theme_counts, theme_freq, severity_df, topic_similarity
