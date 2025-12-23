"""
topic_drift.py

Monitor topic drift over time:
- Fit NMF topics on a reference window (baseline)
- Fit NMF topics on a new window
- Compare topics using Jaccard similarity over top words

Useful for:
- customer feedback monitoring
- social media topic shifts
- model/data drift signals for NLP pipelines

Run:
    python topic_drift.py
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


@dataclass
class DriftConfig:
    n_topics: int = 6
    top_words: int = 15
    max_features: int = 8000
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: Tuple[int, int] = (1, 2)
    random_state: int = 42


def fit_nmf_topics(texts: List[str], cfg: DriftConfig) -> Tuple[NMF, TfidfVectorizer, List[List[str]]]:
    """Fit TF-IDF + NMF and return topic word lists."""
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
    model.fit(X)

    feature_names = np.array(vectorizer.get_feature_names_out())
    H = model.components_  # topic-word matrix

    topic_word_idx = np.argsort(H, axis=1)[:, ::-1]  # descending
    topics = []
    for t in range(cfg.n_topics):
        words = [feature_names[i] for i in topic_word_idx[t, : cfg.top_words]]
        topics.append(words)

    return model, vectorizer, topics


def jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard similarity between two word lists."""
    sa, sb = set(a), set(b)
    denom = len(sa | sb)
    return 0.0 if denom == 0 else len(sa & sb) / denom


def match_topics(
    baseline_topics: List[List[str]],
    new_topics: List[List[str]],
) -> Dict[int, Tuple[int, float]]:
    """
    Match each baseline topic to the most similar new topic.

    Returns:
        {baseline_topic_id: (best_new_topic_id, similarity)}
    """
    matches: Dict[int, Tuple[int, float]] = {}
    for i, base_words in enumerate(baseline_topics):
        sims = [jaccard(base_words, new_words) for new_words in new_topics]
        best_j = int(np.argmax(sims))
        matches[i] = (best_j, float(sims[best_j]))
    return matches


def drift_report(
    baseline_topics: List[List[str]],
    new_topics: List[List[str]],
    low_similarity_threshold: float = 0.25,
) -> List[Dict]:
    """
    Produce a drift report.
    If best similarity is low, that topic is likely drifting / changing.
    """
    matches = match_topics(baseline_topics, new_topics)

    report = []
    for base_id, (new_id, sim) in matches.items():
        report.append(
            {
                "baseline_topic": base_id,
                "new_topic": new_id,
                "similarity": round(sim, 3),
                "drift_flag": sim < low_similarity_threshold,
                "baseline_top_words": baseline_topics[base_id][:10],
                "new_top_words": new_topics[new_id][:10],
            }
        )
    return sorted(report, key=lambda x: x["similarity"])


if __name__ == "__main__":
    # Replace with your real data (e.g., feedback from two different months)
    baseline_docs = [
        "delivery was fast and the package was neat",
        "great price and discount, quality is good",
        "customer service was responsive and helpful",
        "delivery delay and poor communication from courier",
        "quality is excellent and price is fair",
        "late delivery but customer service apologized",
    ]

    new_docs = [
        "delivery delays increased and tracking is inaccurate",
        "refund process is slow and customer support is unhelpful",
        "prices went up but quality is still decent",
        "refund and return policies are confusing",
        "support team not responding, delays everywhere",
        "quality is okay but delivery experience is bad",
    ]

    cfg = DriftConfig(n_topics=3, top_words=12, min_df=1)

    _, _, baseline_topics = fit_nmf_topics(baseline_docs, cfg)
    _, _, new_topics = fit_nmf_topics(new_docs, cfg)

    print("\n=== Baseline Topics ===")
    for i, t in enumerate(baseline_topics):
        print(f"Topic {i}: {', '.join(t[:10])}")

    print("\n=== New Topics ===")
    for i, t in enumerate(new_topics):
        print(f"Topic {i}: {', '.join(t[:10])}")

    report = drift_report(baseline_topics, new_topics, low_similarity_threshold=0.30)

    print("\n=== Drift Report (sorted by similarity) ===")
    for row in report:
        print(
            f"Baseline {row['baseline_topic']} -> New {row['new_topic']} "
            f"(sim={row['similarity']}) drift={row['drift_flag']}\n"
            f"  base: {row['baseline_top_words']}\n"
            f"  new : {row['new_top_words']}\n"
        )
