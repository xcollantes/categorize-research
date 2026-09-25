"""API response expectations for LLM responses."""

from typing import Literal, get_args

from pydantic import BaseModel, Field

Label = Literal["comp.graphics", "rec.sport.baseball", "sci.space", "sci.electronics"]

# Labels are from the dataset newsgroup.
# This will be what the candidates be tested against and will be expected to
# return.
LABEL_DESCRIPTIONS: dict[Label, str] = {
    "comp.graphics": (
        "Computer graphics: rendering, 3D modeling, ray tracing, image "
        "file formats, image processing, and graphics software and "
        "algorithms."
    ),
    "rec.sport.baseball": (
        "Baseball: teams, players, games, pitching and hitting, "
        "statistics, trades, and standings."
    ),
    "sci.space": (
        "Space and spaceflight: NASA, rocket launches, satellites, space "
        "stations, planetary missions, astronomy, and space policy."
    ),
    "sci.electronics": (
        "Electronics: circuit design, electronic components, soldering, "
        "power supplies, amplifiers, radio, and test equipment."
    ),
}
assert LABEL_DESCRIPTIONS.keys() == set(get_args(Label))


class LLMResponse(BaseModel):
    """Expected structure of an LLM classification reply."""

    label: Label = Field(description="The single best-matching topic.")
