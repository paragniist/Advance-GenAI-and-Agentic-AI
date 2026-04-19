"""
Response generation module using Google Gemini.
Handles generating responses from retrieved context.
"""

import time
import random
from typing import List, Dict, Any
from google import genai


class ResponseGenerator:
    """Class for generating responses using Gemini LLM."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.model_name = model
        self.max_retries = max_retries

        # ✅ NEW SDK WAY
        self.client = genai.Client(api_key=self.api_key)

        print(f"🤖 Initialized Gemini Generation Model: {self.model_name}")

    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:

        context = self._build_context(context_chunks)
        prompt = self._create_prompt(query, context)

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )

                return response.text

            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"⏳ Rate limit hit. Waiting {wait_time:.2f}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise Exception("❌ Rate limit exceeded. Please wait and try again.")
                else:
                    print(f"❌ Error generating response: {str(e)}")
                    raise

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:

        if not chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Context {i} - Source: {chunk['source']}, Chunk {chunk['chunk_index']}]\n"
                f"{chunk['text']}\n"
            )

        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:

        prompt = f"""You are a helpful AI assistant answering questions based on provided context.

Context Information:
{context}

User Question: {query}

Instructions:
- Answer the question using ONLY the information provided in the context above
- Be accurate and specific
- If the context doesn't contain enough information to fully answer the question, acknowledge this
- Cite which context sources you used in your answer
- Keep your answer clear and concise

Answer:"""

        return prompt