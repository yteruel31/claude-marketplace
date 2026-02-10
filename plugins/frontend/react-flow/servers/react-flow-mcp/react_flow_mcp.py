#!/usr/bin/env python3
"""
React Flow Documentation MCP Server

MCP server providing React Flow documentation tools. Browse, search, and
retrieve documentation for building node-based UIs with @xyflow/react.

Features:
- List all available documentation pages with optional category filtering
- Get page metadata and available sections
- Retrieve full page content or specific sections
- Search documentation by title, description, or page name

Usage:
    python react_flow_mcp.py
"""

import json
import logging
import re
import sys
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, ConfigDict, Field

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

DATA_DIR = Path(__file__).parent / "data"
DOCS_FILE = DATA_DIR / "docs.json"


# =============================================================================
# Enums
# =============================================================================


class PageCategory(str, Enum):
    API_REFERENCE = "api-reference"
    LEARN = "learn"
    UI = "ui"


# =============================================================================
# Input Models
# =============================================================================


class ListPagesInput(BaseModel):
    """Input for listing documentation pages."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    include_description: bool = Field(
        default=False,
        description="Include page descriptions in the output",
    )
    category: Optional[PageCategory] = Field(
        default=None,
        description="Filter by category: api-reference, learn, or ui",
    )


class GetPageInfoInput(BaseModel):
    """Input for getting page metadata."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    page_name: str = Field(
        ...,
        description=(
            "The name of the page "
            "(e.g., 'api-reference/react-flow', 'learn/concepts/building-a-flow')"
        ),
        min_length=1,
    )


class GetPageInput(BaseModel):
    """Input for getting page content."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    page_name: str = Field(
        ...,
        description=(
            "The name of the page "
            "(e.g., 'api-reference/react-flow', 'learn/concepts/building-a-flow')"
        ),
        min_length=1,
    )
    section_name: Optional[str] = Field(
        default=None,
        description="Optional section name to fetch (use get_page_info to see available sections)",
    )


class SearchDocsInput(BaseModel):
    """Input for searching documentation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description="Search query",
        min_length=1,
    )


# =============================================================================
# Documentation Manager
# =============================================================================


class DocsManager:
    """Manages React Flow documentation data."""

    def __init__(self, docs_path: Path):
        self.docs_path = docs_path
        self.version: str = ""
        self.pages: Dict[str, Dict[str, Any]] = {}

    def load(self) -> None:
        """Load documentation data from JSON file."""
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Documentation file not found: {self.docs_path}")

        with open(self.docs_path, encoding="utf-8") as f:
            data = json.load(f)

        self.version = data.get("version", "unknown")
        self.pages = data.get("pages", {})
        logger.info("Loaded %d documentation pages (version: %s)", len(self.pages), self.version)

    def list_pages(self, include_description: bool = False) -> List[Dict[str, Any]]:
        """List all available documentation pages."""
        result = []
        for page in self.pages.values():
            entry: Dict[str, Any] = {
                "name": page["name"],
                "title": page["title"],
                "category": page["category"],
            }
            if include_description and page.get("description"):
                entry["description"] = page["description"]
            result.append(entry)
        return result

    def get_page_info(self, page_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific page."""
        page = self.pages.get(page_name)
        if not page:
            return None

        return {
            "name": page["name"],
            "title": page["title"],
            "description": page.get("description", ""),
            "category": page["category"],
            "sections": page.get("sections", []),
        }

    def get_page_content(self, page_name: str, section_name: Optional[str] = None) -> Optional[str]:
        """Get the content of a page or a specific section."""
        page = self.pages.get(page_name)
        if not page:
            return None

        content = page.get("content", "")

        if not section_name:
            return content

        # Try h2 section match
        escaped = re.escape(section_name)
        pattern = re.compile(rf"## {escaped}\n([\s\S]*?)(?=\n## |$)", re.IGNORECASE)
        match = pattern.search(content)
        if match:
            return match.group(1).strip()

        # Try h3 section match
        h3_pattern = re.compile(
            rf"### {escaped}\n([\s\S]*?)(?=\n### |\n## |$)", re.IGNORECASE
        )
        h3_match = h3_pattern.search(content)
        if h3_match:
            return h3_match.group(1).strip()

        return None

    def search_pages(self, query: str) -> List[Dict[str, Any]]:
        """Search pages by title, description, or page name."""
        lower_query = query.lower()
        results = []

        for page in self.pages.values():
            if (
                lower_query in page["title"].lower()
                or lower_query in page.get("description", "").lower()
                or lower_query in page["name"].lower()
            ):
                results.append({
                    "name": page["name"],
                    "title": page["title"],
                    "category": page["category"],
                    "description": page.get("description", ""),
                })

        return results


# =============================================================================
# Global state
# =============================================================================

docs_manager: Optional[DocsManager] = None


# =============================================================================
# Lifespan
# =============================================================================


@asynccontextmanager
async def app_lifespan(app):
    """Initialize documentation data on startup."""
    global docs_manager

    logger.info("Initializing React Flow documentation server...")

    docs_manager = DocsManager(DOCS_FILE)
    try:
        docs_manager.load()
    except FileNotFoundError as e:
        logger.error("Failed to load docs: %s", e)
        logger.error("Run 'uv run python fetch_docs.py' to fetch documentation first.")
        raise

    logger.info("React Flow MCP server ready")
    yield {"docs_manager": docs_manager}
    logger.info("React Flow MCP server shutting down")


# =============================================================================
# MCP Server
# =============================================================================

mcp = FastMCP("react-flow", lifespan=app_lifespan)


# =============================================================================
# Tools
# =============================================================================


@mcp.tool(
    name="list_pages",
    annotations={
        "title": "List React Flow Documentation Pages",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_pages(params: ListPagesInput, ctx: Context) -> str:
    """Returns a list of available pages in the React Flow docs.
    Use this to discover what documentation is available."""
    logger.info("list_pages called (category=%s)", params.category)

    pages = docs_manager.list_pages(include_description=params.include_description)

    if params.category:
        pages = [p for p in pages if p["category"] == params.category.value]

    logger.info("Returning %d pages", len(pages))
    return json.dumps(pages, indent=2)


@mcp.tool(
    name="get_page_info",
    annotations={
        "title": "Get React Flow Page Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_page_info(params: GetPageInfoInput, ctx: Context) -> str:
    """Returns page description and list of sections for a given page.
    Use this to understand what a page contains before fetching full content."""
    logger.info("get_page_info called (page=%s)", params.page_name)

    info = docs_manager.get_page_info(params.page_name)

    if not info:
        error_msg = (
            f'Page not found: {params.page_name}. '
            f'Use list_pages to see available pages.'
        )
        logger.warning(error_msg)
        return json.dumps({"error": error_msg})

    return json.dumps(info, indent=2)


@mcp.tool(
    name="get_page",
    annotations={
        "title": "Get React Flow Page Content",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_page(params: GetPageInput, ctx: Context) -> str:
    """Returns the full markdown content for a page, or a specific section if provided."""
    logger.info("get_page called (page=%s, section=%s)", params.page_name, params.section_name)

    content = docs_manager.get_page_content(params.page_name, params.section_name)

    if content is None:
        if params.section_name:
            error_msg = (
                f'Section "{params.section_name}" not found in page "{params.page_name}". '
                f'Use get_page_info to see available sections.'
            )
        else:
            error_msg = (
                f'Page not found: {params.page_name}. '
                f'Use list_pages to see available pages.'
            )
        logger.warning(error_msg)
        return json.dumps({"error": error_msg})

    return content


@mcp.tool(
    name="search_docs",
    annotations={
        "title": "Search React Flow Documentation",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def search_docs(params: SearchDocsInput, ctx: Context) -> str:
    """Search React Flow documentation by title, description, or page name."""
    logger.info("search_docs called (query=%s)", params.query)

    results = docs_manager.search_pages(params.query)

    if not results:
        return json.dumps({"message": f'No pages found matching "{params.query}".'})

    logger.info("Found %d matching pages", len(results))
    return json.dumps(results, indent=2)


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Run the MCP server."""
    logger.info("Starting React Flow MCP server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
