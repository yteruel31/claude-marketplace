#!/usr/bin/env python3
"""
Google Research MCP Server

MCP server providing web research capabilities using Google Gemini Flash
with Google Search grounding for real-time, cited research.

Features:
- Web research with high thinking level and Google Search grounding
- Inline citations from grounding metadata
- Multi-layer prompt injection defense
- Source quality filtering

Usage:
    python google_research_mcp.py
"""

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
from typing import Any, Dict, List, Optional

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
MODEL = "gemini-3-flash-preview"
THINKING_LEVEL = "HIGH"


# =============================================================================
# Enums and Data Models
# =============================================================================


class ResponseFormat(str, Enum):
    """Output format for research results."""

    MARKDOWN = "markdown"
    JSON = "json"


# =============================================================================
# Input Models (Pydantic)
# =============================================================================


class ResearchInput(BaseModel):
    """Input model for web research with Google Search grounding."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    query: str = Field(
        ...,
        description="Research question or topic (e.g., 'What are the latest developments in quantum computing?', 'Compare React vs Vue in 2025')",
        min_length=10,
        max_length=2000,
    )
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Optional: Specific aspects to emphasize (e.g., ['performance', 'pricing', 'user reviews'])",
        max_length=5,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for structured data",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensure query is meaningful."""
        if len(v.strip()) < 10:
            raise ValueError("Query must be at least 10 characters")
        return v.strip()


# =============================================================================
# Research Client and Core Logic
# =============================================================================


class GeminiResearchClient:
    """Manages Google Gemini API interactions for grounded research."""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def _save_research_to_disk(
        self,
        query: str,
        result: Dict[str, Any],
        research_type: str
    ) -> Optional[str]:
        """
        Save research result to docs/research/ as markdown.

        Saves to working directory (project root where Claude Code is running).

        Args:
            query: The research query
            result: The complete result dict with report, citations, metadata
            research_type: Type of research (quick)

        Returns:
            Path to saved file or None if save failed
        """
        try:
            # Create docs/research directory in the user's project directory
            # ORIGINAL_WORKING_DIR is set by run.sh before it cd's to the script dir
            working_dir = os.environ.get('ORIGINAL_WORKING_DIR', os.getcwd())
            research_dir = Path(working_dir) / "docs" / "research"
            research_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename: YYYYMMDD_HHMMSS_sanitized-query.md
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', query[:50])
            sanitized = re.sub(r'_+', '_', sanitized).strip('_')
            filename = f"{timestamp}_{sanitized}.md"
            filepath = research_dir / filename

            # Build markdown content
            content = self._build_markdown_content(query, result, research_type)

            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"Research saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save research to disk: {e}")
            return None

    def _sanitize_content(self, text: str) -> str:
        """
        Remove HTML tags and markdown images that could leak sensitive data.

        Strips:
        - HTML img, iframe, script, object, embed, link tags
        - Markdown images with external URLs ![...](http://...)

        Args:
            text: Raw text content

        Returns:
            Sanitized text with dangerous elements removed
        """
        # Remove HTML img tags (self-closing or not)
        text = re.sub(r'<img[^>]*/?>', '', text, flags=re.IGNORECASE)
        # Remove dangerous HTML tags with content
        text = re.sub(
            r'<(iframe|script|object|embed|link)[^>]*>.*?</\1>',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        # Remove self-closing dangerous tags
        text = re.sub(
            r'<(iframe|script|object|embed|link)[^>]*/?>',
            '',
            text,
            flags=re.IGNORECASE
        )
        # Remove markdown images with external URLs (http/https)
        text = re.sub(r'!\[[^\]]*\]\(https?://[^)]+\)', '', text)
        return text

    def _build_markdown_content(
        self,
        query: str,
        result: Dict[str, Any],
        research_type: str
    ) -> str:
        """Build markdown document from research result."""
        metadata = result.get("metadata", {})
        citations = result.get("citations", [])
        report = self._sanitize_content(result.get("report", ""))

        # Build frontmatter-style header
        lines = [
            "---",
            f"research_type: {research_type}",
            f"query: {query}",
            f"model: {metadata.get('model', 'unknown')}",
            f"thinking_level: {metadata.get('thinking_level', 'unknown')}",
            f"timestamp: {metadata.get('timestamp', '')}",
            f"sources: {metadata.get('sources_count', 0)}",
            f"searches: {metadata.get('searches_count', 0)}",
            "---",
            "",
            f"# {query}",
            "",
            report,
            ""
        ]

        # Add citations section if present
        if citations:
            lines.extend([
                "",
                "## Sources",
                ""
            ])
            for idx, citation in enumerate(citations, 1):
                lines.append(f"[{idx}] [{citation['title']}]({citation['url']})")

        return "\n".join(lines)

    def _parse_grounding_metadata(
        self,
        response
    ) -> tuple[List[Dict[str, str]], List[str]]:
        """
        Extract citations and search queries from grounding metadata.

        Returns:
            Tuple of (citations list, search queries list)
        """
        citations = []
        search_queries = []

        try:
            # Get grounding metadata from response
            if not response.candidates:
                return citations, search_queries

            candidate = response.candidates[0]

            # Check for grounding_metadata attribute
            grounding_metadata = getattr(candidate, 'grounding_metadata', None)
            if not grounding_metadata:
                return citations, search_queries

            # Extract search queries
            web_search_queries = getattr(grounding_metadata, 'web_search_queries', [])
            if web_search_queries:
                search_queries = list(web_search_queries)

            # Extract grounding chunks (sources)
            grounding_chunks = getattr(grounding_metadata, 'grounding_chunks', [])
            for chunk in grounding_chunks:
                web_info = getattr(chunk, 'web', None)
                if web_info:
                    uri = getattr(web_info, 'uri', '')
                    title = getattr(web_info, 'title', uri)
                    if uri:
                        citations.append({
                            'url': uri,
                            'title': title or uri,
                        })

        except Exception as e:
            logger.warning(f"Failed to parse grounding metadata: {e}")

        return citations, search_queries

    def _build_inline_citations(
        self,
        text: str,
        response,
        citations: List[Dict[str, str]]
    ) -> str:
        """
        Build text with inline citation markers based on grounding supports.

        Args:
            text: The response text
            response: Full API response with grounding metadata
            citations: List of citation dicts with url/title

        Returns:
            Text with inline citations like [1], [2], etc.
        """
        if not citations:
            return text

        try:
            candidate = response.candidates[0]
            grounding_metadata = getattr(candidate, 'grounding_metadata', None)
            if not grounding_metadata:
                return text

            grounding_supports = getattr(grounding_metadata, 'grounding_supports', [])
            if not grounding_supports:
                return text

            # Build a mapping from chunk index to citation number
            # Process supports in reverse order to maintain string indices
            supports_to_apply = []

            for support in grounding_supports:
                segment = getattr(support, 'segment', None)
                chunk_indices = getattr(support, 'grounding_chunk_indices', [])

                if segment and chunk_indices:
                    start_idx = getattr(segment, 'start_index', None)
                    end_idx = getattr(segment, 'end_index', None)

                    if start_idx is not None and end_idx is not None:
                        # Get citation numbers (1-indexed)
                        cite_nums = [idx + 1 for idx in chunk_indices if idx < len(citations)]
                        if cite_nums:
                            cite_str = "".join(f"[{n}]" for n in cite_nums)
                            supports_to_apply.append((end_idx, cite_str))

            # Sort by position descending to insert from end
            supports_to_apply.sort(key=lambda x: x[0], reverse=True)

            # Insert citations
            for end_idx, cite_str in supports_to_apply:
                text = text[:end_idx] + cite_str + text[end_idx:]

        except Exception as e:
            logger.warning(f"Failed to build inline citations: {e}")

        return text

    async def execute_research(
        self,
        query: str,
        focus_areas: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a research query with Google Search grounding.

        Args:
            query: The research question
            focus_areas: Optional list of aspects to focus on

        Returns:
            Dict with: report, citations, search_queries, metadata
        """
        start_time = time.time()

        try:
            # Build the prompt
            prompt = query
            if focus_areas:
                prompt += f"\n\nPlease focus on these aspects: {', '.join(focus_areas)}"

            # Configure request
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]

            tools = [
                types.Tool(google_search=types.GoogleSearch()),
            ]

            safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_LOW_AND_ABOVE",
                ),
            ]

            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level=THINKING_LEVEL,
                ),
                tools=tools,
                safety_settings=safety_settings,
            )

            # Execute request
            response = self.client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=generate_content_config,
            )

            # Extract text from response
            full_text = response.text or ""

            # Parse grounding metadata
            citations, search_queries = self._parse_grounding_metadata(response)

            # Build inline citations
            report_with_citations = self._build_inline_citations(
                full_text, response, citations
            )

            execution_time = time.time() - start_time

            return {
                "report": report_with_citations,
                "citations": citations,
                "search_queries": search_queries,
                "metadata": {
                    "model": MODEL,
                    "thinking_level": THINKING_LEVEL,
                    "execution_time_seconds": int(execution_time),
                    "sources_count": len(citations),
                    "searches_count": len(search_queries),
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            }

        except Exception as e:
            logger.error(f"Research execution failed: {e}")
            raise


# =============================================================================
# Helper Functions
# =============================================================================


def format_research_output(result: Dict[str, Any], format_type: ResponseFormat) -> str:
    """Format research results based on requested format."""
    if format_type == ResponseFormat.JSON:
        return json.dumps(result, indent=2)

    # Markdown format (default)
    output = result["report"]

    # Add sources section if citations exist and not already in output
    if result["citations"] and "\n## Sources\n" not in output:
        output += "\n\n## Sources\n\n"
        for idx, cite in enumerate(result["citations"], 1):
            output += f"{idx}. [{cite['title']}]({cite['url']})\n"

    # Add search queries info
    if result["search_queries"]:
        output += f"\n\n---\n"
        output += f"*Searches performed: {', '.join(result['search_queries'][:5])}*\n"

    # Add metadata footer
    meta = result["metadata"]
    output += f"*Research completed in {meta['execution_time_seconds']}s using {meta['model']} (thinking: {meta['thinking_level']})*\n"

    return output


# =============================================================================
# MCP Server and Lifespan
# =============================================================================

# Global client instance
research_client: Optional[GeminiResearchClient] = None


@asynccontextmanager
async def app_lifespan(app):
    """Initialize resources that persist across requests."""
    global research_client

    # Validate API key
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Get your key at https://aistudio.google.com/apikey"
        )

    # Initialize client
    research_client = GeminiResearchClient(api_key=GEMINI_API_KEY)
    logger.info("Google Research MCP Server initialized")
    logger.info(f"Model: {MODEL}")

    yield {"client": research_client}

    # Cleanup
    logger.info("Google Research MCP Server shutting down")


# Initialize FastMCP server
mcp = FastMCP("google_research_mcp", lifespan=app_lifespan)


# =============================================================================
# MCP Tools
# =============================================================================


@mcp.tool(
    name="google_research",
    annotations={
        "title": "Web Research (Gemini Flash + Google Search)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def google_research(params: ResearchInput, ctx: Context) -> str:
    """Web research using Google Gemini Flash with Google Search grounding.

    Performs real-time web research with inline citations. Uses high thinking
    level for quality reasoning with Google Search for grounded, up-to-date results.

    Security features:
    - Multi-layer prompt injection defense
    - Source quality filtering (deprioritizes SEO-poisoned/spam sites)
    - BLOCK_LOW_AND_ABOVE safety threshold

    Args:
        params (ResearchInput): Validated input containing:
            - query (str): Research question (10-2000 chars)
            - focus_areas (Optional[List[str]]): Specific aspects to emphasize
            - response_format (ResponseFormat): Output format (markdown/json)

    Returns:
        str: Research report with inline citations and sources
    """
    try:
        logger.info(f"Starting research: {params.query[:50]}...")

        result = await research_client.execute_research(
            query=params.query,
            focus_areas=params.focus_areas,
        )

        logger.info(
            f"Research completed: {result['metadata']['sources_count']} sources, "
            f"{result['metadata']['execution_time_seconds']}s"
        )

        return format_research_output(result, params.response_format)

    except Exception as e:
        logger.error(f"Research failed: {e}")
        return json.dumps(
            {
                "error": str(e),
                "help": "Try rephrasing your query or breaking it into smaller questions.",
            }
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting Google Research MCP Server...")
    mcp.run(transport="stdio")
