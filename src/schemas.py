"""
Pydantic schemas and request/response models for the Sarcasm Detection API.
"""

from pydantic import BaseModel, Field, field_validator


class TextInput(BaseModel):
    """
    Schema for the incoming text prediction request.
    Enforces strict input length constraints and sanitization to prevent DOS/memory exhaustion.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="News headline text to analyze for sarcasm (1 to 500 characters).",
        examples=["Man Finally Finishes Reading Terms and Conditions Agreement"],
    )

    @field_validator("text", mode="before")
    @classmethod
    def validate_and_sanitize_text(cls, value: object) -> str:
        """
        Sanitizes text by stripping surrounding whitespace and rejecting empty/blank inputs.
        """
        if not isinstance(value, str):
            raise ValueError("Input text must be a valid string.")

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Text cannot be empty or contain only whitespace.")

        return cleaned


class PredictionResult(BaseModel):
    """Result details for a single sarcasm prediction."""

    text: str = Field(..., description="Analyzed input text.")
    prediction: str = Field(..., description="Classification label: SARCASTIC or GENUINE.")
    confidence: float = Field(..., description="Confidence score percentage (0.0 to 100.0).")
    is_sarcastic: bool = Field(..., description="True if text is classified as sarcastic.")


class PredictionResponse(BaseModel):
    """Standard API response wrapper for prediction results."""

    status: str = Field(default="success", description="Status code of the operation.")
    data: PredictionResult = Field(..., description="Prediction payload.")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Overall health status: healthy or degraded.")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded and ready.")
    device: str | None = Field(default=None, description="Computation device (cpu or cuda).")
