"""
Simple LLM Prompt Template Utility
----------------------------------

Purpose:
- Show how to structure prompts for Large Language Models (LLMs).
- This file is API-agnostic: you can use it with OpenAI, Anthropic, etc.
- Safe to commit publicly (no keys, no secrets).

Usage idea:
- Import `build_ds_assistant_prompt()` in another script or notebook.
- Send the returned string to your chosen LLM client.
"""

from textwrap import dedent


def build_ds_assistant_prompt(user_question: str) -> str:
    """
    Build a structured prompt for a Data Science / AI assistant.

    Parameters
    ----------
    user_question : str
        The question or task you want the LLM to answer.

    Returns
    -------
    str
        A well-structured prompt string.
    """

    system_block = dedent(
        """
        You are an expert Data Science and Machine Learning assistant.
        - Explain concepts clearly and step-by-step.
        - When writing code, use clean, commented examples.
        - If there are multiple valid approaches, briefly compare them.
        - If information is uncertain, say so instead of guessing.
        """
    ).strip()

    instructions_block = dedent(
        """
        When answering:
        - Start with a direct answer.
        - Then provide a short explanation.
        - If relevant, include a minimal runnable code example.
        - Avoid unnecessary jargon.
        """
    ).strip()

    user_block = f"User question:\n{user_question.strip()}"

    prompt = f"{system_block}\n\n{instructions_block}\n\n{user_block}\n\nAssistant:"
    return prompt


if __name__ == "__main__":
    # Example usage (for local testing):
    example_question = "Explain the difference between random forest and gradient boosting."
    print(build_ds_assistant_prompt(example_question))
