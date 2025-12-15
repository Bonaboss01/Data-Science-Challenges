"""
Topic modelling utilities using:
- TF-IDF + NMF (recommended for short/medium text)
- CountVectorizer + LDA (classic baseline)

Run:
    python topic_modeling.py
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


@dataclass
class TopicModelConfig:
    n_topics: int = 8
    top_words: int = 12
    max_features: int = 5000
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: Tuple[int, int] = (1, 2)
    random_state: int = 42


def _format_topics(feature_names: np.ndarray, topic_word_idx: np.ndarray, top_words: int) -> List[List[str]]:
    topics = []
    for t in range(topic_word_idx.shape[0]):
        words = [feature_names[i] for i in topic_word_idx[t, :top_words]]
        topics.append(words)
    return topics


def nmf_topics(texts: List[str], cfg: TopicModelConfig) -> Tuple[NMF, TfidfVectorizer, List[List[str]], np.ndarray]:
    """
    Fit an NMF topic model on TF-IDF features.

    Returns:
        model, vectorizer, topics(list of words), doc_topic_matrix
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        ngram_range=cfg.ngram_range,
    )
    X = vectorizer.fit_transform(texts)

    model = NMF(
        n_components=cfg.n_topics,
        init="nndsvda",
        random_state=cfg.random_state,
        max_iter=400,
    )
    W = model.fit_transform(X)  # document-topic matrix
    H = model.components_       # topic-word matrix

    feature_names = np.array(vectorizer.get_feature_names_out())
    topic_word_idx = np.argsort(H, axis=1)[:, ::-1]  # descending
    topics = _format_topics(feature_names, topic_word_idx, cfg.top_words)

    return model, vectorizer, topics, W


def lda_topics(texts: List[str], cfg: TopicModelConfig) -> Tuple[LatentDirichletAllocation, CountVectorizer, List[List[str]], np.ndarray]:
    """
    Fit an LDA topic model on CountVectorizer features.

    Returns:
        model, vectorizer, topics(list of words), doc_topic_matrix
    """
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        ngram_range=cfg.ngram_range,
    )
    X = vectorizer.fit_transform(texts)

    model = LatentDirichletAllocation(
        n_components=cfg.n_topics,
        random_state=cfg.random_state,
        learning_method="batch",
        max_iter=30,
    )
    doc_topic = model.fit_transform(X)

    feature_names = np.array(vectorizer.get_feature_names_out())
    topic_word_idx = np.argsort(model.components_, axis=1)[:, ::-1]
    topics = _format_topics(feature_names, topic_word_idx, cfg.top_words)

    return model, vectorizer, topics, doc_topic


def dominant_topic(doc_topic_matrix: np.ndarray) -> np.ndarray:
    """Return dominant topic index per document."""
    return np.argmax(doc_topic_matrix, axis=1)


def topic_distribution(doc_topic_matrix: np.ndarray) -> Dict[int, int]:
    """Count how many docs fall into each dominant topic."""
    dom = dominant_topic(doc_topic_matrix)
    counts = {int(k): int(v) for k, v in zip(*np.unique(dom, return_counts=True))}
    return counts


if __name__ == "__main__":
    # Example dataset (replace with your own texts)
    documents = [
        "Universal Credit forecasting uses time series models and driver comparisons.",
        "Topic modelling helps discover themes in customer feedback and survey comments.",
        "NLP pipelines include cleaning, tokenization, embeddings, and classifiers.",
        "Forecast accuracy improves with better feature engineering and validation.",
        "LDA and NMF are common topic modelling approaches for text analytics.",
        "We deployed dashboards to communicate insights to stakeholders.",
        "We used TF-IDF features to build interpretable models for text data.",
        "Seasonality and trend components matter in time series forecasting.",
        "Stakeholders want clear narratives and actionable recommendations.",
        "Customer reviews often mention price, delivery, quality, and service."
    ]

    cfg = TopicModelConfig(n_topics=4, top_words=8, min_df=1)

    print("\n=== NMF (TF-IDF) Topics ===")
    nmf_model, tfidf_vec, nmf_topic_words, nmf_doc_topic = nmf_topics(documents, cfg)
    for i, words in enumerate(nmf_topic_words):
        print(f"Topic {i}: {', '.join(words)}")
    print("Dominant topic distribution:", topic_distribution(nmf_doc_topic))

    print("\n=== LDA (Counts) Topics ===")
    lda_model, count_vec, lda_topic_words, lda_doc_topic = lda_topics(documents, cfg)
    for i, words in enumerate(lda_topic_words):
        print(f"Topic {i}: {', '.join(words)}")
    print("Dominant topic distribution:", topic_distribution(lda_doc_topic))
