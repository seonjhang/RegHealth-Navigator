"""
summarizer.py

Module for summarizing pre-chunked regulatory text extracted from XML documents.

Expected input:
    chunks: List[str]  # List of chunked text extracted from XML

Output:
    str  # Final combined summary

Example:
>>> summarize_chunks(['chunk1 text', 'chunk2 text'])
'Final combined summary of the regulation.'
"""

from typing import List
import openai

# Prompt template for per-chunk summarization (Chain of Thought)
CHUNK_PROMPT = """
The following text is a portion of a U.S. government healthcare regulation.

Please summarize it by answering the following 4 questions:
1. What is the main purpose of this section?
2. What policy changes or updates are described?
3. Which stakeholders are affected?
4. What actions or responses are expected from them?

Provide the response as a coherent paragraph based on those four points.

[Start of text]
{chunk}
[End of text]
"""

# Prompt for final summary synthesis
FINAL_SUMMARY_PROMPT = """
Below are summaries of individual sections of a U.S. healthcare regulation document.

Please combine them into a single, cohesive executive summary that captures the core points and major updates of the full document.

Section Summaries:
{summaries}
"""

def summarize_chunks(chunks: List[str]) -> str:
    """
    Summarize a list of regulatory text chunks using an LLM with Chain of Thought reasoning.

    Args:
        chunks (List[str]): List of text chunks.

    Returns:
        str: Final combined summary.
    """
    individual_summaries = []

    for i, chunk in enumerate(chunks):
        prompt = CHUNK_PROMPT.format(chunk=chunk)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            summary = response.choices[0].message.content.strip()
            individual_summaries.append(summary)
            print(f"✅ Chunk {i+1}/{len(chunks)} summarized")
        except Exception as e:
            print(f"❌ Error summarizing chunk {i+1}: {e}")
            continue

    # Combine all individual summaries into a final executive summary
    joined = "\n\n".join(individual_summaries)
    final_prompt = FINAL_SUMMARY_PROMPT.format(summaries=joined)

    try:
        final_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.3
        )
        return final_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error generating final summary: {e}")
        return joined  # Fallback: return joined summaries
