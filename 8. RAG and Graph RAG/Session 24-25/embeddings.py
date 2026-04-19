"""
Embedding generation module using Google Gemini.
Handles text-to-vector conversion with retry logic.
"""

import time
import random
from typing import List
from google import genai


class EmbeddingGenerator:
    """Class for generating embeddings using Gemini API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",   # ⚠️ No "models/"
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries

        # Create client (NEW SDK way)
        self.client = genai.Client(api_key=self.api_key)

        print(f"🤖 Initialized Gemini Embedding Model: {self.model}")

    def generate(self, text: str, task_type: str = "retrieval_document") -> List[float]:

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                )

                return response.embeddings[0].values

            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"⏳ Rate limit hit. Waiting {wait_time:.2f}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise Exception("❌ Rate limit exceeded. Please wait and try again.")
                else:
                    print(f"❌ Error generating embedding: {str(e)}")
                    raise

    def generate_query_embedding(self, query: str) -> List[float]:
        return self.generate(query)

    def generate_document_embedding(self, document: str) -> List[float]:
        return self.generate(document)