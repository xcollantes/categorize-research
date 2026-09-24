"""API response expectations for LLM responses."""

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Class representing the expected structure of an LLM response."""

    output_text: str = Field(description="The generated text output from the LLM.")
