#!/usr/bin/env python3
"""
Google Image MCP Server

MCP server providing AI image generation using Google Gemini.
Images are saved to disk and the path returned for inline display in Claude Code.

Features:
- Text-to-image generation via Gemini API
- Configurable output directory, aspect ratio, and resolution
- Automatic file saving with timestamped filenames
- Returns absolute paths for Claude Code Read tool integration

Usage:
    python google_image_mcp.py
"""

import base64
import json
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Configure logging to stderr only (required for stdio transport)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration and Constants
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = "gemini-2.5-flash-image"
DEFAULT_OUTPUT_DIR = "generated-images"


# =============================================================================
# Enums and Data Models
# =============================================================================


class AspectRatio(str, Enum):
    """Supported aspect ratios for image generation."""

    SQUARE = "1:1"
    LANDSCAPE_WIDE = "16:9"
    PORTRAIT_TALL = "9:16"
    LANDSCAPE = "4:3"
    PORTRAIT = "3:4"


class ImageModel(str, Enum):
    """Available Gemini models for image generation."""

    FLASH = "gemini-2.5-flash-image"
    PRO = "gemini-3-pro-image-preview"


class ImageSize(str, Enum):
    """Output resolution tier."""

    ONE_K = "1K"
    TWO_K = "2K"
    FOUR_K = "4K"


# =============================================================================
# Input Models (Pydantic)
# =============================================================================


class GenerateImageInput(BaseModel):
    """Input model for image generation."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    prompt: str = Field(
        ...,
        description=(
            "Text description of the image to generate "
            "(e.g., 'A watercolor painting of a cat sitting on a windowsill')"
        ),
        min_length=5,
        max_length=2000,
    )
    output_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory to save the generated image. "
            "Relative paths are resolved from the project root. "
            "Absolute paths are used as-is. "
            "Defaults to 'generated-images/'."
        ),
    )
    image_size: ImageSize = Field(
        default=ImageSize.ONE_K,
        description="Output resolution: '1K' (default, ~1024px), '2K' (~2048px), '4K' (ultra HD)",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="Image aspect ratio: '1:1' (default), '16:9', '9:16', '4:3', '3:4'",
    )
    model: ImageModel = Field(
        default=ImageModel.FLASH,
        description=(
            "Gemini model to use: "
            "'gemini-2.5-flash-image' (default, fast and cost-efficient) or "
            "'gemini-3-pro-image-preview' (higher quality, reasoning-driven)"
        ),
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Ensure prompt is meaningful."""
        if len(v.strip()) < 5:
            raise ValueError("Prompt must be at least 5 characters")
        return v.strip()


# =============================================================================
# Image Generation Client
# =============================================================================


class GeminiImageClient:
    """Manages Google Gemini API interactions for image generation."""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def _get_output_dir(self, output_dir: Optional[str] = None) -> Path:
        """Get the output directory for generated images.

        Uses ORIGINAL_WORKING_DIR (set by run.sh) to resolve relative paths
        from the user's project directory.

        Args:
            output_dir: Custom output directory. If None, uses DEFAULT_OUTPUT_DIR.
                        Absolute paths are used as-is, relative paths are resolved
                        from the project root.
        """
        target = output_dir or DEFAULT_OUTPUT_DIR
        path = Path(target)

        if not path.is_absolute():
            working_dir = os.environ.get("ORIGINAL_WORKING_DIR", os.getcwd())
            path = Path(working_dir) / path

        path.mkdir(parents=True, exist_ok=True)
        return path

    def _generate_filename(self, prompt: str) -> str:
        """Generate a timestamped filename from the prompt.

        Pattern: YYYYMMDD_HHMMSS_sanitized-prompt.png
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", prompt[:50])
        sanitized = re.sub(r"_+", "_", sanitized).strip("_").lower()
        return f"{timestamp}_{sanitized}.png"

    def _save_image(self, image_data: bytes, prompt: str, output_dir: Optional[str] = None) -> str:
        """Save image bytes to disk and return the absolute path.

        Args:
            image_data: Raw PNG bytes
            prompt: Original prompt (used for filename)
            output_dir: Custom output directory

        Returns:
            Absolute path to saved file
        """
        dir_path = self._get_output_dir(output_dir)
        filename = self._generate_filename(prompt)
        filepath = dir_path / filename

        with open(filepath, "wb") as f:
            f.write(image_data)

        logger.info(f"Image saved to {filepath}")
        return str(filepath.resolve())

    async def generate_image(
        self,
        prompt: str,
        output_dir: Optional[str] = None,
        image_size: ImageSize = ImageSize.ONE_K,
        aspect_ratio: AspectRatio = AspectRatio.SQUARE,
        model: str = DEFAULT_MODEL,
    ) -> dict:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image
            output_dir: Custom output directory
            image_size: Output resolution tier
            aspect_ratio: Desired aspect ratio
            model: Gemini model to use

        Returns:
            Dict with: file_path, metadata
        """
        start_time = time.time()

        try:
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    image_size=image_size.value,
                ),
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_LOW_AND_ABOVE",
                    ),
                ],
            )

            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            # Extract image from response
            image_data = None
            text_response = None

            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        image_bytes = part.inline_data.data
                        if isinstance(image_bytes, str):
                            image_data = base64.b64decode(image_bytes)
                        else:
                            image_data = image_bytes
                    elif part.text is not None:
                        text_response = part.text

            if image_data is None:
                error_msg = (
                    "No image was returned by the API. "
                    "The model may have refused the prompt or encountered an error."
                )
                if text_response:
                    error_msg += f" Model response: {text_response}"
                raise ValueError(error_msg)

            # Save to disk
            file_path = self._save_image(image_data, prompt, output_dir)

            execution_time = time.time() - start_time

            return {
                "file_path": file_path,
                "metadata": {
                    "model": model,
                    "image_size": image_size.value,
                    "aspect_ratio": aspect_ratio.value,
                    "prompt": prompt,
                    "execution_time_seconds": round(execution_time, 1),
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "file_size_bytes": len(image_data),
                },
            }

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise


# =============================================================================
# MCP Server and Lifespan
# =============================================================================

# Global client instance
image_client: Optional[GeminiImageClient] = None


@asynccontextmanager
async def app_lifespan(app):
    """Initialize resources that persist across requests."""
    global image_client

    # Validate API key
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Get your key at https://aistudio.google.com/apikey"
        )

    # Initialize client
    image_client = GeminiImageClient(api_key=GEMINI_API_KEY)
    logger.info("Google Image MCP Server initialized")
    logger.info(f"Default model: {DEFAULT_MODEL}")

    yield {"client": image_client}

    # Cleanup
    logger.info("Google Image MCP Server shutting down")


# Initialize FastMCP server
mcp = FastMCP("google_image_mcp", lifespan=app_lifespan)


# =============================================================================
# MCP Tools
# =============================================================================


@mcp.tool(
    name="generate_image",
    annotations={
        "title": "Generate Image (Gemini)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def generate_image(params: GenerateImageInput, ctx: Context) -> str:
    """Generate an image from a text prompt using Google Gemini.

    Creates an AI-generated image and saves it as a PNG file. Returns the
    absolute file path so Claude Code can display it inline using the Read tool.

    Images are saved to the configured output directory (default:
    `generated-images/` in the user's working directory) with timestamped
    filenames.

    Args:
        params (GenerateImageInput): Validated input containing:
            - prompt (str): Image description (5-2000 chars)
            - output_dir (Optional[str]): Custom output directory
            - image_size (ImageSize): Output resolution (1K, 2K, 4K)
            - aspect_ratio (AspectRatio): Image dimensions ratio
            - model (ImageModel): Gemini model to use

    Returns:
        str: JSON with file_path and metadata. Use the Read tool on file_path
             to display the image inline.
    """
    try:
        logger.info(f"Generating image: {params.prompt[:50]}...")

        result = await image_client.generate_image(
            prompt=params.prompt,
            output_dir=params.output_dir,
            image_size=params.image_size,
            aspect_ratio=params.aspect_ratio,
            model=params.model.value,
        )

        logger.info(
            f"Image generated: {result['file_path']} "
            f"({result['metadata']['file_size_bytes']} bytes, "
            f"{result['metadata']['execution_time_seconds']}s)"
        )

        output = {
            "file_path": result["file_path"],
            "message": (
                f"Image saved to: {result['file_path']}\n"
                "Use the Read tool on this file path to display the image inline."
            ),
            "metadata": result["metadata"],
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return json.dumps(
            {
                "error": str(e),
                "help": "Try simplifying your prompt or using a different model.",
            }
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting Google Image MCP Server...")
    mcp.run(transport="stdio")
