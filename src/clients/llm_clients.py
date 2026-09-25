"""Client for interacting with LLMs."""

import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

from src.models.response_models import Label, LLMResponse

logger: logging.Logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]


class GPTClient:
    """Client for interacting with OpenAI (ChatGPT) models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client: OpenAI = OpenAI(api_key=OPENAI_API_KEY)

    def classify(self, prompt: str) -> Label | None:
        """Classify the prompt into one label; None if the model refuses."""
        response = self.client.responses.parse(
            model=self.model_name,
            input=prompt,
            text_format=LLMResponse,
        )

        parsed: LLMResponse | None = response.output_parsed
        return parsed.label if parsed else None


class GeminiClient:
    """Client for interacting with Gemini LLMs."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client: genai.Client = genai.Client(api_key=GEMINI_API_KEY)

    def classify(self, prompt: str) -> Label | None:
        """Classify the prompt into one label; None if the model refuses."""
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
        return parsed.label if parsed else None
