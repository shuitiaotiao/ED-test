from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class UploadedEntry(BaseModel):
    path: str = Field(..., min_length=1, description="Relative file path or a single zip archive name.")
    content_base64: str = Field(..., min_length=1, description="Base64-encoded file contents.")


class ValidationRequest(BaseModel):
    files: list[UploadedEntry] = Field(..., min_length=1)
    label: str | None = Field(default=None, max_length=80)
    bc_y: Literal["PBC", "APBC"] = "PBC"
    max_basis_states: int = Field(default=250_000, ge=1, le=10_000_000)

