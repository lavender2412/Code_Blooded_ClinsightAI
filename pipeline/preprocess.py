import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [LEMMATIZER.lemmatize(w) for w in words if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)

def clean_dataframe(df, text_col="Feedback", rating_col="Ratings"):
    """Load, clean and return dataframe with clean_text column."""
    import pandas as pd
    df = df[[text_col, rating_col]].dropna()
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    df = df.dropna(subset=[rating_col]).reset_index(drop=True)
    df[rating_col] = df[rating_col].astype(float)
    df["clean_text"] = df[text_col].apply(preprocess)
    return df