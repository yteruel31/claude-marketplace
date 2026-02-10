#!/usr/bin/env python3
"""
React Flow Documentation Fetcher

Fetches React Flow documentation from the xyflow/web GitHub repository
and bundles it into a JSON file for the MCP server.

Usage:
    python fetch_docs.py
    # or via pyproject.toml script:
    uv run fetch-docs
"""

import asyncio
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

GITHUB_API = "https://api.github.com"
REPO = "xyflow/web"
BASE_PATH = "sites/reactflow.dev/src/content"
USER_AGENT = "react-flow-mcp"

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "docs.json"

# Directories to fetch
DIRECTORIES = [
    # API Reference
    "api-reference",
    "api-reference/components",
    "api-reference/hooks",
    "api-reference/types",
    "api-reference/utils",
    # Learn
    "learn",
    "learn/advanced-use",
    "learn/concepts",
    "learn/customization",
    "learn/layouting",
    "learn/troubleshooting",
    "learn/tutorials",
    # UI
    "ui",
    "ui/components",
    "ui/templates",
]

# Rate limit delay (seconds) between file fetches
FETCH_DELAY = 0.1


# =============================================================================
# Parsing Utilities
# =============================================================================


def parse_frontmatter(content: str) -> tuple[Dict[str, str], str]:
    """Parse YAML frontmatter from MDX content.

    Returns (frontmatter_dict, body_content).
    """
    match = re.match(r"^---\n([\s\S]*?)\n---", content)
    if not match:
        return {}, content.strip()

    frontmatter: Dict[str, str] = {}
    for line in match.group(1).split("\n"):
        kv = re.match(r"^(\w+):\s*(.*)$", line)
        if kv:
            value = kv.group(2).strip()
            # Remove surrounding quotes
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            frontmatter[kv.group(1)] = value

    body = content[match.end() :].strip()
    return frontmatter, body


def extract_sections(content: str) -> List[str]:
    """Extract section titles (h2 and h3) from markdown content."""
    sections = []
    for line in content.split("\n"):
        h2 = re.match(r"^## (.+)$", line)
        h3 = re.match(r"^### (.+)$", line)
        if h2:
            sections.append(h2.group(1))
        elif h3:
            sections.append(h3.group(1))
    return sections


def path_to_page_name(path: str) -> str:
    """Convert a GitHub file path to a page name."""
    name = path.replace(f"{BASE_PATH}/", "").replace(".mdx", "")
    # Remove trailing /index
    name = re.sub(r"/index$", "", name)
    return name


def determine_category(directory: str) -> str:
    """Determine page category from its directory path."""
    if directory.startswith("learn"):
        return "learn"
    elif directory.startswith("ui"):
        return "ui"
    return "api-reference"


# =============================================================================
# GitHub API Client
# =============================================================================


async def fetch_directory(client: httpx.AsyncClient, path: str) -> List[Dict[str, Any]]:
    """Fetch directory contents from the GitHub API."""
    url = f"{GITHUB_API}/repos/{REPO}/contents/{BASE_PATH}/{path}"
    logger.info("Fetching: %s", path)

    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error("Failed to fetch %s: %s", path, e.response.status_code)
        return []
    except httpx.RequestError as e:
        logger.error("Request error for %s: %s", path, e)
        return []


async def fetch_file_content(client: httpx.AsyncClient, download_url: str) -> Optional[str]:
    """Fetch file content from a GitHub download URL."""
    try:
        response = await client.get(download_url)
        response.raise_for_status()
        return response.text
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        logger.error("Failed to fetch file: %s", e)
        return None


# =============================================================================
# Main
# =============================================================================


async def fetch_all_docs() -> Dict[str, Any]:
    """Fetch all React Flow documentation from GitHub."""
    docs: Dict[str, Any] = {
        "version": datetime.now(timezone.utc).isoformat(),
        "pages": {},
    }

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": USER_AGENT,
    }

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        for directory in DIRECTORIES:
            contents = await fetch_directory(client, directory)

            for item in contents:
                if item.get("type") != "file" or not item.get("name", "").endswith(".mdx"):
                    continue

                page_name = path_to_page_name(item["path"])
                logger.info("  Processing: %s", page_name)

                content = await fetch_file_content(client, item["download_url"])
                if not content:
                    logger.warning("    Failed to fetch content for %s", page_name)
                    continue

                frontmatter, body = parse_frontmatter(content)
                sections = extract_sections(body)
                category = determine_category(directory)

                title = (
                    frontmatter.get("title")
                    or frontmatter.get("sidebarTitle")
                    or item["name"].replace(".mdx", "")
                )

                docs["pages"][page_name] = {
                    "name": page_name,
                    "title": title,
                    "description": frontmatter.get("description", ""),
                    "category": category,
                    "sections": sections,
                    "content": body,
                }

                # Small delay to avoid rate limiting
                await asyncio.sleep(FETCH_DELAY)

    return docs


def main():
    """Main entry point for the documentation fetcher."""
    logger.info("Fetching React Flow documentation...")

    docs = asyncio.run(fetch_all_docs())

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)

    page_count = len(docs["pages"])
    logger.info("Done! Fetched %d pages.", page_count)
    logger.info("Output: %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
