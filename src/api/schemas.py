"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field


class LineResult(BaseModel):
    """OCR result for a single detected line."""
    line: int = Field(..., description="Line number (1-indexed)")
    raw_text: str = Field(..., description="Raw OCR output")
    clean_text: str = Field(..., description="NLP-corrected text")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


class ModelStatus(BaseModel):
    """Status of loaded ML models."""
    trocr: bool = Field(False, description="TrOCR model loaded")
    crnn: bool = Field(False, description="CRNN model loaded")
    indobert: bool = Field(False, description="IndoBERT model loaded")
    lexicon: bool = Field(False, description="Lexicon loaded")


class OCRResponse(BaseModel):
    """Response schema for the /ocr endpoint."""
    raw_text: str = Field(..., description="Full raw OCR text")
    clean_text: str = Field(..., description="Full NLP-corrected text")
    confidence: float = Field(..., ge=0, le=1, description="Average confidence")
    lines_detected: int = Field(..., ge=0, description="Number of text lines")
    lines: list[LineResult] = Field(default_factory=list)
    models_used: ModelStatus = Field(default_factory=ModelStatus)


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""
    status: str = Field("healthy")
    models: ModelStatus = Field(default_factory=ModelStatus)
    device: str = Field("cpu")


class RootResponse(BaseModel):
    """Response schema for the root endpoint."""
    service: str = Field("NoteScanin AI")
    version: str = Field("2.0.0")
    pipeline: list[str] = Field(default_factory=list)
    endpoints: dict[str, str] = Field(default_factory=dict)
