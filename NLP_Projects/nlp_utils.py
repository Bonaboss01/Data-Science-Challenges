"""
Reusable NLP utilities for text cleaning, tokenization,
and lightweight feature extraction.
"""

import re
from typing import List, Dict
from collections import Counter


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Lowercase
    - Remove punctuation
    - Remove extra whitespace
    """

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize cleaned text into words.
    """
    return text.split(" ")


def word_frequency(tokens: List[str], top_n: int = 10) -> Dict[str, int]:
    """
    Return the most common words.

    Args:
        tokens (list): List of tokens
        top_n (int): Number of top words

    Returns:
        dict
    """
    return dict(Counter(tokens).most_common(top_n))


def text_to_features(text: str) -> Dict[str, float]:
    """
    Extract simple numerical features from text.
    Useful for baseline NLP models.

    Returns:
        dict
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)

    return {
        "char_count": len(text),
        "word_count": len(tokens),
        "unique_word_count": len(set(tokens)),
        "avg_word_length": sum(len(t) for t in tokens) / max(len(tokens), 1)
    }


if __name__ == "__main__":
    sample_text = "NLP is powerful! NLP helps machines understand text."

    cleaned = clean_text(sample_text)
    tokens = tokenize(cleaned)

    print("Cleaned text:", cleaned)
    print("Tokens:", tokens)
    print("Top words:", word_frequency(tokens))
    print("Text features:", text_to_features(sample_text))
