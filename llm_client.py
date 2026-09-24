"""Client for interacting with LLMs."""

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

logger: logging.Logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]


class LLMClient:
    """Client for interacting with OpenAI (ChatGPT) models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client: OpenAI = OpenAI(api_key=OPENAI_API_KEY)

    def generate_text(self, prompt: str) -> str:
        """Generate text based on the given prompt."""

        response = self.client.responses.create(model=self.model_name, input=prompt)
        return response.output_text
