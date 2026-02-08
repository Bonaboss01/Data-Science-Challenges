"""
text.py

Basic text analysis:
- Count characters
- Count words
- Count sentences
"""

def analyze_text(text):
    characters = len(text)
    words = len(text.split())
    sentences = text.count(".") + text.count("!") + text.count("?")

    return {
        "characters": characters,
        "words": words,
        "sentences": sentences
    }


if __name__ == "__main__":
    sample = "Data science is powerful. Python makes it easier!"
    result = analyze_text(sample)

    print("Text Analysis")
    for k, v in result.items():
        print(f"{k}: {v}")
