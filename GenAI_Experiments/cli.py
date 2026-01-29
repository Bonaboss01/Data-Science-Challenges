import argparse

from src.genai.rag_pipeline import RAGConfig, answer_question


def main():
    parser = argparse.ArgumentParser(description="Simple LangChain RAG CLI")
    parser.add_argument("question", type=str, help="Your question")
    parser.add_argument("--docs_dir", default="data/knowledge/docs")
    parser.add_argument("--faiss_dir", default="data/knowledge/faiss_index")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    cfg = RAGConfig(
        docs_dir=args.docs_dir,
        faiss_dir=args.faiss_dir,
        top_k=args.top_k,
        model=args.model,
    )

    answer, sources = answer_question(args.question, cfg)
    print("\nANSWER:\n", answer)
    print("\nSOURCES:\n", ", ".join(sources) if sources else "None")


if __name__ == "__main__":
    main()
