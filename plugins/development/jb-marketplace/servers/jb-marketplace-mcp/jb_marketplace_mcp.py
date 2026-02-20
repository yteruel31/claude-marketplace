#!/usr/bin/env python3
"""
JetBrains Marketplace MCP Server

MCP server providing read-only access to the JetBrains Marketplace API.

Features:
- Search plugins by keyword
- Get plugin metadata by numeric ID
- List plugin version history
- Check plugin compatibility with IDE builds

Usage:
    python jb_marketplace_mcp.py
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
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
# Configuration
# =============================================================================

API_BASE = "https://plugins.jetbrains.com"

# =============================================================================
# Input Models
# =============================================================================


class SearchPluginsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(..., description="Search query", min_length=1)
    build: Optional[str] = Field(
        default=None,
        description="IDE build number to filter compatibility (e.g., 'IC-251.25410.129')",
    )
    max: int = Field(default=10, description="Max results to return (1-100)", ge=1, le=100)


class GetPluginInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plugin_id: int = Field(..., description="Numeric plugin ID from JetBrains Marketplace")


class ListUpdatesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plugin_id: int = Field(..., description="Numeric plugin ID")
    size: int = Field(default=10, description="Number of updates to return (1-100)", ge=1, le=100)
    channel: Optional[str] = Field(
        default=None,
        description="Release channel filter (e.g., 'Stable', 'EAP'). Omit for all channels.",
    )


class CheckCompatibleInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    build: str = Field(
        ..., description="IDE build number (e.g., 'IC-251.25410.129')", min_length=1
    )
    plugin_xml_ids: List[str] = Field(
        ..., description="List of plugin XML IDs to check", min_length=1
    )


# =============================================================================
# Global State
# =============================================================================

http_client: Optional[httpx.AsyncClient] = None

# =============================================================================
# Lifespan and Server Init
# =============================================================================


@asynccontextmanager
async def app_lifespan(app):
    """Initialize the HTTP client for the Marketplace API."""
    global http_client
    http_client = httpx.AsyncClient(base_url=API_BASE, timeout=30.0)
    logger.info("JetBrains Marketplace MCP Server initialized")

    yield {"http": http_client}

    await http_client.aclose()
    logger.info("JetBrains Marketplace MCP Server shut down")


mcp = FastMCP("jb_marketplace_mcp", lifespan=app_lifespan)


# =============================================================================
# Tools
# =============================================================================


@mcp.tool(
    name="search_plugins",
    annotations={
        "title": "Search JetBrains Marketplace Plugins",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def search_plugins(params: SearchPluginsInput, ctx: Context) -> str:
    """Search JetBrains Marketplace plugins by keyword.

    Args:
        params: SearchPluginsInput with query, optional build filter, and max results

    Returns:
        JSON string with matching plugins
    """
    try:
        query_params: dict = {"search": params.query, "max": params.max}
        if params.build:
            query_params["build"] = params.build

        response = await http_client.get("/api/search/plugins", params=query_params)
        response.raise_for_status()
        plugins = response.json()
        return json.dumps(
            {"results": plugins, "count": len(plugins), "query": params.query},
            indent=2,
        )
    except Exception as e:
        logger.error(f"Failed to search plugins: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="get_plugin",
    annotations={
        "title": "Get JetBrains Plugin Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def get_plugin(params: GetPluginInput, ctx: Context) -> str:
    """Get detailed metadata for a JetBrains Marketplace plugin.

    Args:
        params: GetPluginInput with numeric plugin_id

    Returns:
        JSON string with plugin metadata
    """
    try:
        response = await http_client.get(f"/api/plugins/{params.plugin_id}")
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return json.dumps({"error": f"Plugin {params.plugin_id} not found"})
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error(f"Failed to get plugin: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="list_updates",
    annotations={
        "title": "List JetBrains Plugin Updates",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def list_updates(params: ListUpdatesInput, ctx: Context) -> str:
    """List version history and updates for a JetBrains Marketplace plugin.

    Args:
        params: ListUpdatesInput with plugin_id, size, and optional channel

    Returns:
        JSON string with version history
    """
    try:
        query_params: dict = {"size": params.size}
        if params.channel:
            query_params["channel"] = params.channel

        response = await http_client.get(
            f"/api/plugins/{params.plugin_id}/updates", params=query_params
        )
        response.raise_for_status()
        updates = response.json()
        return json.dumps(
            {"updates": updates, "count": len(updates), "plugin_id": params.plugin_id},
            indent=2,
        )
    except Exception as e:
        logger.error(f"Failed to list updates: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="check_compatible",
    annotations={
        "title": "Check JetBrains Plugin Compatibility",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def check_compatible(params: CheckCompatibleInput, ctx: Context) -> str:
    """Check if plugins are compatible with a specific IDE build.

    Args:
        params: CheckCompatibleInput with build number and plugin XML IDs

    Returns:
        JSON string with compatible update info for each plugin
    """
    try:
        response = await http_client.post(
            "/api/search/compatibleUpdates",
            json={"build": params.build, "pluginXMLIds": params.plugin_xml_ids},
        )
        response.raise_for_status()
        results = response.json()
        return json.dumps(
            {"results": results, "count": len(results), "build": params.build},
            indent=2,
        )
    except Exception as e:
        logger.error(f"Failed to check compatibility: {e}")
        return json.dumps({"error": str(e)})


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting JetBrains Marketplace MCP Server...")
    mcp.run(transport="stdio")
