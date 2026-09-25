"""Client for classifying text with Gemini embeddings."""

import logging
import math
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

logger: logging.Logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]

# gemini-embedding-2 takes the task as a text prefix, not a task_type config.
# https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2
CLASSIFICATION_PREFIX: str = "task: classification | query: "


class EmbedClient:
    """Client for Gemini embedding models."""

    def __init__(self, model_name: str = "gemini-embedding-2"):
        self.model_name = model_name
        self.client: genai.Client = genai.Client(api_key=GEMINI_API_KEY)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each text for classification, one vector per text.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per text.
        """

        contents = [
            types.Content(parts=[types.Part(text=CLASSIFICATION_PREFIX + t)])
            for t in texts
        ]

        result = self.client.models.embed_content(
            model=self.model_name, contents=contents
        )

        return [e.values for e in result.embeddings]

    def classify(self, text: str, label_vecs: dict[str, list[float]]) -> str:
        """Return the label whose embedding is closest to the text.

        Args:
            text: The text to classify.
            label_vecs: Label to its precomputed embedding, so labels are
                embedded once per run instead of once per text.

        Returns:
            The label whose embedding is closest to the text.
        """

        (text_vec,) = self.embed([text])

        return max(label_vecs, key=lambda k: _cosine(text_vec, label_vecs[k]))

    def count_tokens(self, text: str) -> int:
        """Return the billed input tokens for embedding `text`."""
        result = self.client.models.count_tokens(
            model=self.model_name, contents=CLASSIFICATION_PREFIX + text
        )

        return result.total_tokens or 0


def _cosine(a: list[float], b: list[float]) -> float:
    return math.sumprod(a, b) / (math.hypot(*a) * math.hypot(*b))
