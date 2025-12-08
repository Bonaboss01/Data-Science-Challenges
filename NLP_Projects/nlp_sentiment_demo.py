# nlp_sentiment_demo.py
# Simple NLP sentiment analysis demo using NLTK VADER
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER if missing
nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

sample_texts = [
    "I absolutely love this product!",
    "This is the worst thing I have ever bought.",
    "It's okay, not great but not terrible either."
]

def analyse_sentiment(text):
    score = sia.polarity_scores(text)
    return score["compound"]

for t in sample_texts:
    print(f"Text: {t}")
    print(f"Sentiment Score: {analyse_sentiment(t)}")
    print("-" * 40)
