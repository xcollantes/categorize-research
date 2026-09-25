"""A candidate classifier and the list prices it is billed at."""

from pydantic import BaseModel, ConfigDict, Field


class Candidate(BaseModel):
    """One experimental condition: a named model and its token prices."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Key in results.jsonl and the summary.")
    model_id: str = Field(description="Provider model ID sent to the API.")
    input_usd_per_million_tok: float = Field(ge=0, description="USD per 1M input.")
    output_usd_per_million_tok: float = Field(ge=0, description="USD per 1M output.")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return the USD list-price cost of one call."""
        return (
            input_tokens * self.input_usd_per_million_tok
            + output_tokens * self.output_usd_per_million_tok
        ) / 1e6
