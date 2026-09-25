"""Client for interacting with LLMs."""

import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

from src.models.response_models import LLMResponse, Prediction

logger: logging.Logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]


class GPTClient:
    """Client for interacting with OpenAI (ChatGPT) models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client: OpenAI = OpenAI(api_key=OPENAI_API_KEY)

    def classify(self, prompt: str) -> Prediction:
        """Classify the prompt into one label; label is None on refusal."""
        response = self.client.responses.parse(
            model=self.model_name,
            input=prompt,
            text_format=LLMResponse,
        )

        parsed: LLMResponse | None = response.output_parsed
        usage = response.usage
        # Reasoning tokens are already included in output_tokens.
        return Prediction(
            parsed.label if parsed else None,
            usage.input_tokens if usage else 0,
            usage.output_tokens if usage else 0,
        )


class GeminiClient:
    """Client for interacting with Gemini LLMs."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client: genai.Client = genai.Client(api_key=GEMINI_API_KEY)

    def classify(self, prompt: str) -> Prediction:
        """Classify the prompt into one label; label is None on refusal."""
        response: types.GenerateContentResponse = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                # To ensure 100% valid JSON objects, requests must include both
                # a response_schema and response_mime_type: "application/json".
                response_mime_type="application/json",
                response_schema=LLMResponse,
            ),
        )

        parsed: LLMResponse | None = response.parsed
        usage = (
            response.usage_metadata
            or types.GenerateContentResponseUsageMetadata()
        )
        # Thinking tokens are billed as output but counted separately.
        return Prediction(
            parsed.label if parsed else None,
            usage.prompt_token_count or 0,
            (usage.candidates_token_count or 0)
            + (usage.thoughts_token_count or 0),
        )
