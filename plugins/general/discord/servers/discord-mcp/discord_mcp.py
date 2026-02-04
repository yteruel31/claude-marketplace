#!/usr/bin/env python3
"""
Discord MCP Server

MCP server providing Discord bot integration and API documentation access.

Features:
- Send messages to Discord channels
- Read message history
- Search messages by content
- List guilds and channels
- Fetch Discord API documentation from GitHub

Usage:
    python discord_mcp.py
"""

import asyncio
import json
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, List, Optional

import discord
import httpx
from discord import Intents
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

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_DOCS_BASE_URL = "https://raw.githubusercontent.com/discord/discord-api-docs/main/docs"
DISCORD_DOCS_API_URL = "https://api.github.com/repos/discord/discord-api-docs/git/trees/main?recursive=1"

MAX_MESSAGE_LENGTH = 2000
DEFAULT_MESSAGE_LIMIT = 25
MAX_MESSAGE_LIMIT = 100

# Documentation sections available
DOC_SECTIONS = [
    "activities",
    "change-log",
    "components",
    "developer-tools",
    "discord-social-sdk",
    "discovery",
    "events",
    "interactions",
    "monetization",
    "policies-and-agreements",
    "quick-start",
    "resources",
    "rich-presence",
    "topics",
    "tutorials",
]


# =============================================================================
# Enums and Input Models
# =============================================================================


class ChannelType(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    CATEGORY = "category"
    ALL = "all"


class SendMessageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    channel_id: str = Field(..., description="Discord channel ID to send message to")
    content: str = Field(
        ..., description="Message content (max 2000 chars)", max_length=MAX_MESSAGE_LENGTH
    )
    reply_to_message_id: Optional[str] = Field(
        default=None, description="Optional message ID to reply to"
    )


class ReadMessagesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel_id: str = Field(..., description="Discord channel ID to read from")
    limit: int = Field(
        default=DEFAULT_MESSAGE_LIMIT,
        description="Number of messages to retrieve (1-100)",
        ge=1,
        le=MAX_MESSAGE_LIMIT,
    )


class ListChannelsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guild_id: str = Field(..., description="Discord guild/server ID")
    channel_type: ChannelType = Field(
        default=ChannelType.ALL,
        description="Filter by channel type: text, voice, category, or all",
    )


class SearchMessagesInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    channel_id: str = Field(..., description="Channel ID to search in")
    query: str = Field(..., description="Search query (case-insensitive)", min_length=1)
    limit: int = Field(
        default=50, description="Max messages to search through", ge=1, le=MAX_MESSAGE_LIMIT
    )


class DocsListInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section: Optional[str] = Field(
        default=None,
        description=f"Optional section to list files from (e.g., 'interactions', 'resources'). Available: {', '.join(DOC_SECTIONS)}",
    )


class DocsSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(..., description="Search query (case-insensitive)", min_length=2)
    limit: int = Field(default=10, description="Max results to return", ge=1, le=50)


class DocsFetchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    path: str = Field(
        ...,
        description="Path to doc file (e.g., 'interactions/application_commands.mdx', 'resources/channel.mdx')",
    )


# =============================================================================
# Discord Client Manager
# =============================================================================


class DiscordClientManager:
    """Manages the Discord client lifecycle and operations."""

    def __init__(self, token: str):
        self.token = token
        self._client: Optional[discord.Client] = None
        self._ready_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    @property
    def client(self) -> discord.Client:
        if self._client is None:
            raise RuntimeError("Discord client not initialized")
        return self._client

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    async def start(self):
        """Start the Discord client as a background task."""
        intents = Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True

        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready():
            logger.info(f"Discord bot connected as {self._client.user}")
            logger.info(f"Connected to {len(self._client.guilds)} guilds")
            self._ready_event.set()

        # Start client in background
        self._task = asyncio.create_task(self._client.start(self.token))

        # Wait for ready with timeout
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Discord client failed to connect within 30 seconds")

    async def stop(self):
        """Stop the Discord client gracefully."""
        if self._client:
            await self._client.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def send_message(
        self, channel_id: str, content: str, reply_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a message to a channel."""
        channel = await self.client.fetch_channel(int(channel_id))

        reference = None
        if reply_to:
            reference = discord.MessageReference(
                message_id=int(reply_to), channel_id=int(channel_id)
            )

        message = await channel.send(content, reference=reference)

        return {
            "message_id": str(message.id),
            "channel_id": str(message.channel.id),
            "content": message.content,
            "timestamp": message.created_at.isoformat(),
            "author": str(self.client.user),
        }

    async def read_messages(self, channel_id: str, limit: int) -> List[Dict[str, Any]]:
        """Read recent messages from a channel."""
        channel = await self.client.fetch_channel(int(channel_id))

        messages = []
        async for msg in channel.history(limit=limit):
            messages.append(
                {
                    "message_id": str(msg.id),
                    "author": str(msg.author),
                    "author_id": str(msg.author.id),
                    "content": msg.content,
                    "timestamp": msg.created_at.isoformat(),
                    "attachments": [a.url for a in msg.attachments],
                    "reactions": [
                        {"emoji": str(r.emoji), "count": r.count} for r in msg.reactions
                    ],
                }
            )

        return messages

    async def list_guilds(self) -> List[Dict[str, Any]]:
        """List all guilds the bot is in."""
        return [
            {
                "guild_id": str(guild.id),
                "name": guild.name,
                "member_count": guild.member_count,
                "owner_id": str(guild.owner_id),
                "icon_url": str(guild.icon.url) if guild.icon else None,
            }
            for guild in self.client.guilds
        ]

    async def list_channels(
        self, guild_id: str, channel_type: ChannelType
    ) -> List[Dict[str, Any]]:
        """List channels in a guild."""
        guild = await self.client.fetch_guild(int(guild_id))
        channels = await guild.fetch_channels()

        result = []
        for ch in channels:
            ch_type = None
            if isinstance(ch, discord.TextChannel):
                ch_type = "text"
            elif isinstance(ch, discord.VoiceChannel):
                ch_type = "voice"
            elif isinstance(ch, discord.CategoryChannel):
                ch_type = "category"

            if channel_type != ChannelType.ALL and ch_type != channel_type.value:
                continue

            result.append(
                {
                    "channel_id": str(ch.id),
                    "name": ch.name,
                    "type": ch_type,
                    "position": ch.position,
                    "category": ch.category.name if ch.category else None,
                }
            )

        return sorted(result, key=lambda x: (x["category"] or "", x["position"]))

    async def get_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """Get detailed information about a channel."""
        channel = await self.client.fetch_channel(int(channel_id))

        info = {
            "channel_id": str(channel.id),
            "name": channel.name,
            "type": str(channel.type),
            "created_at": channel.created_at.isoformat(),
        }

        if isinstance(channel, discord.TextChannel):
            info.update(
                {
                    "topic": channel.topic,
                    "slowmode_delay": channel.slowmode_delay,
                    "nsfw": channel.is_nsfw(),
                    "category": channel.category.name if channel.category else None,
                    "guild_id": str(channel.guild.id),
                    "guild_name": channel.guild.name,
                }
            )

        return info

    async def get_guild_info(self, guild_id: str) -> Dict[str, Any]:
        """Get detailed information about a guild."""
        guild = await self.client.fetch_guild(int(guild_id))

        return {
            "guild_id": str(guild.id),
            "name": guild.name,
            "description": guild.description,
            "member_count": guild.member_count,
            "owner_id": str(guild.owner_id),
            "created_at": guild.created_at.isoformat(),
            "icon_url": str(guild.icon.url) if guild.icon else None,
            "banner_url": str(guild.banner.url) if guild.banner else None,
            "premium_tier": guild.premium_tier,
            "premium_subscription_count": guild.premium_subscription_count,
        }

    async def search_messages(
        self, channel_id: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Search messages in a channel by content."""
        channel = await self.client.fetch_channel(int(channel_id))
        query_lower = query.lower()

        results = []
        async for msg in channel.history(limit=limit):
            if query_lower in msg.content.lower():
                results.append(
                    {
                        "message_id": str(msg.id),
                        "author": str(msg.author),
                        "content": msg.content,
                        "timestamp": msg.created_at.isoformat(),
                    }
                )

        return results


# =============================================================================
# Documentation Client
# =============================================================================


class DocsClient:
    """Manages fetching Discord API documentation from GitHub."""

    def __init__(self):
        self._http_client: Optional[httpx.AsyncClient] = None
        self._docs_index: Optional[List[Dict[str, str]]] = None

    async def start(self):
        """Initialize the HTTP client."""
        self._http_client = httpx.AsyncClient(timeout=30.0)

    async def stop(self):
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")
        return self._http_client

    async def _fetch_docs_index(self) -> List[Dict[str, str]]:
        """Fetch the list of all documentation files from GitHub."""
        if self._docs_index is not None:
            return self._docs_index

        try:
            response = await self.http.get(DISCORD_DOCS_API_URL)
            response.raise_for_status()
            data = response.json()

            # Filter for .md and .mdx files in docs/
            docs_files = []
            for item in data.get("tree", []):
                path = item.get("path", "")
                if path.startswith("docs/") and (
                    path.endswith(".md") or path.endswith(".mdx")
                ):
                    # Remove docs/ prefix
                    relative_path = path[5:]
                    section = relative_path.split("/")[0] if "/" in relative_path else ""
                    docs_files.append(
                        {
                            "path": relative_path,
                            "section": section,
                            "name": os.path.basename(relative_path),
                        }
                    )

            self._docs_index = docs_files
            return docs_files

        except Exception as e:
            logger.error(f"Failed to fetch docs index: {e}")
            raise

    async def list_docs(self, section: Optional[str] = None) -> List[Dict[str, str]]:
        """List available documentation files."""
        docs = await self._fetch_docs_index()

        if section:
            docs = [d for d in docs if d["section"] == section]

        return docs

    async def search_docs(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documentation files by filename and content."""
        docs = await self._fetch_docs_index()
        query_lower = query.lower()

        # First, filter by filename
        results = []
        for doc in docs:
            if query_lower in doc["name"].lower() or query_lower in doc["path"].lower():
                results.append(
                    {
                        "path": doc["path"],
                        "section": doc["section"],
                        "name": doc["name"],
                        "match_type": "filename",
                    }
                )

        # If we don't have enough results, search content
        if len(results) < limit:
            for doc in docs[:50]:  # Limit content search to first 50 files
                if any(r["path"] == doc["path"] for r in results):
                    continue

                try:
                    content = await self.fetch_doc(doc["path"])
                    if query_lower in content.lower():
                        # Find matching line for context
                        lines = content.split("\n")
                        context = ""
                        for line in lines:
                            if query_lower in line.lower():
                                context = line[:100].strip()
                                break

                        results.append(
                            {
                                "path": doc["path"],
                                "section": doc["section"],
                                "name": doc["name"],
                                "match_type": "content",
                                "context": context,
                            }
                        )

                        if len(results) >= limit:
                            break

                except Exception:
                    continue

        return results[:limit]

    async def fetch_doc(self, path: str) -> str:
        """Fetch a specific documentation file."""
        # Clean path
        path = path.lstrip("/")
        if not path.endswith((".md", ".mdx")):
            path += ".mdx"

        url = f"{DISCORD_DOCS_BASE_URL}/{path}"

        try:
            response = await self.http.get(url)
            response.raise_for_status()
            return response.text

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Documentation not found: {path}")
            raise


# =============================================================================
# Global Instances
# =============================================================================

discord_manager: Optional[DiscordClientManager] = None
docs_client: Optional[DocsClient] = None


# =============================================================================
# MCP Server and Lifespan
# =============================================================================


@asynccontextmanager
async def app_lifespan(app):
    """Initialize resources that persist across requests."""
    global discord_manager, docs_client

    # Always initialize docs client (no auth required)
    docs_client = DocsClient()
    await docs_client.start()
    logger.info("Documentation client initialized")

    # Initialize Discord client if token is provided
    if DISCORD_BOT_TOKEN:
        discord_manager = DiscordClientManager(DISCORD_BOT_TOKEN)
        await discord_manager.start()
        logger.info("Discord client connected")
    else:
        logger.warning(
            "DISCORD_BOT_TOKEN not set. Bot tools will not be available. "
            "Documentation tools will still work."
        )

    logger.info("Discord MCP Server initialized")

    yield {"discord": discord_manager, "docs": docs_client}

    # Cleanup
    if discord_manager:
        await discord_manager.stop()
    if docs_client:
        await docs_client.stop()

    logger.info("Discord MCP Server shut down")


# Initialize FastMCP server
mcp = FastMCP("discord_mcp", lifespan=app_lifespan)


# =============================================================================
# Bot Tools (require DISCORD_BOT_TOKEN)
# =============================================================================


def _require_discord_client() -> DiscordClientManager:
    """Helper to check if Discord client is available."""
    if discord_manager is None or not discord_manager.is_ready:
        raise RuntimeError(
            "Discord bot not connected. Set DISCORD_BOT_TOKEN environment variable."
        )
    return discord_manager


@mcp.tool(
    name="send_message",
    annotations={
        "title": "Send Discord Message",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def send_message(params: SendMessageInput, ctx: Context) -> str:
    """Send a message to a Discord channel.

    Args:
        params: SendMessageInput containing channel_id, content, and optional reply_to_message_id

    Returns:
        JSON string with sent message details
    """
    try:
        dm = _require_discord_client()
        result = await dm.send_message(
            channel_id=params.channel_id,
            content=params.content,
            reply_to=params.reply_to_message_id,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="read_messages",
    annotations={
        "title": "Read Discord Messages",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def read_messages(params: ReadMessagesInput, ctx: Context) -> str:
    """Read recent messages from a Discord channel.

    Args:
        params: ReadMessagesInput containing channel_id and limit

    Returns:
        JSON string with list of messages
    """
    try:
        dm = _require_discord_client()
        messages = await dm.read_messages(
            channel_id=params.channel_id,
            limit=params.limit,
        )
        return json.dumps({"messages": messages, "count": len(messages)}, indent=2)
    except Exception as e:
        logger.error(f"Failed to read messages: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="search_messages",
    annotations={
        "title": "Search Discord Messages",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def search_messages(params: SearchMessagesInput, ctx: Context) -> str:
    """Search for messages containing specific text in a Discord channel.

    Args:
        params: SearchMessagesInput containing channel_id, query, and limit

    Returns:
        JSON string with matching messages
    """
    try:
        dm = _require_discord_client()
        results = await dm.search_messages(
            channel_id=params.channel_id,
            query=params.query,
            limit=params.limit,
        )
        return json.dumps(
            {"results": results, "count": len(results), "query": params.query}, indent=2
        )
    except Exception as e:
        logger.error(f"Failed to search messages: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="list_guilds",
    annotations={
        "title": "List Discord Guilds",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def list_guilds(ctx: Context) -> str:
    """List all Discord guilds (servers) the bot is a member of.

    Returns:
        JSON string with list of guilds
    """
    try:
        dm = _require_discord_client()
        guilds = await dm.list_guilds()
        return json.dumps({"guilds": guilds, "count": len(guilds)}, indent=2)
    except Exception as e:
        logger.error(f"Failed to list guilds: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="list_channels",
    annotations={
        "title": "List Discord Channels",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def list_channels(params: ListChannelsInput, ctx: Context) -> str:
    """List all channels in a Discord guild.

    Args:
        params: ListChannelsInput containing guild_id and optional channel_type filter

    Returns:
        JSON string with list of channels
    """
    try:
        dm = _require_discord_client()
        channels = await dm.list_channels(
            guild_id=params.guild_id,
            channel_type=params.channel_type,
        )
        return json.dumps({"channels": channels, "count": len(channels)}, indent=2)
    except Exception as e:
        logger.error(f"Failed to list channels: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="get_channel_info",
    annotations={
        "title": "Get Discord Channel Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def get_channel_info(channel_id: str, ctx: Context) -> str:
    """Get detailed information about a Discord channel.

    Args:
        channel_id: The Discord channel ID

    Returns:
        JSON string with channel details
    """
    try:
        dm = _require_discord_client()
        info = await dm.get_channel_info(channel_id)
        return json.dumps(info, indent=2)
    except Exception as e:
        logger.error(f"Failed to get channel info: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="get_guild_info",
    annotations={
        "title": "Get Discord Guild Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def get_guild_info(guild_id: str, ctx: Context) -> str:
    """Get detailed information about a Discord guild (server).

    Args:
        guild_id: The Discord guild ID

    Returns:
        JSON string with guild details
    """
    try:
        dm = _require_discord_client()
        info = await dm.get_guild_info(guild_id)
        return json.dumps(info, indent=2)
    except Exception as e:
        logger.error(f"Failed to get guild info: {e}")
        return json.dumps({"error": str(e)})


# =============================================================================
# Documentation Tools (no auth required)
# =============================================================================


@mcp.tool(
    name="docs_list",
    annotations={
        "title": "List Discord API Docs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def docs_list(params: DocsListInput, ctx: Context) -> str:
    """List available Discord API documentation sections and files.

    Args:
        params: DocsListInput with optional section filter

    Returns:
        JSON string with list of documentation files
    """
    try:
        docs = await docs_client.list_docs(section=params.section)

        if params.section:
            return json.dumps(
                {"section": params.section, "files": docs, "count": len(docs)}, indent=2
            )
        else:
            # Group by section
            sections = {}
            for doc in docs:
                section = doc["section"] or "root"
                if section not in sections:
                    sections[section] = []
                sections[section].append(doc["name"])

            return json.dumps(
                {
                    "available_sections": DOC_SECTIONS,
                    "sections": sections,
                    "total_files": len(docs),
                },
                indent=2,
            )

    except Exception as e:
        logger.error(f"Failed to list docs: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="docs_search",
    annotations={
        "title": "Search Discord API Docs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def docs_search(params: DocsSearchInput, ctx: Context) -> str:
    """Search Discord API documentation by keyword.

    Searches both filenames and content for matches.

    Args:
        params: DocsSearchInput with query and limit

    Returns:
        JSON string with search results
    """
    try:
        results = await docs_client.search_docs(query=params.query, limit=params.limit)
        return json.dumps(
            {"query": params.query, "results": results, "count": len(results)}, indent=2
        )
    except Exception as e:
        logger.error(f"Failed to search docs: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="docs_fetch",
    annotations={
        "title": "Fetch Discord API Doc",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def docs_fetch(params: DocsFetchInput, ctx: Context) -> str:
    """Fetch the content of a specific Discord API documentation page.

    Args:
        params: DocsFetchInput with path to the doc file

    Returns:
        The raw markdown/mdx content of the documentation page
    """
    try:
        content = await docs_client.fetch_doc(path=params.path)
        return content
    except ValueError as e:
        return json.dumps({"error": str(e), "hint": "Use docs_list to see available files"})
    except Exception as e:
        logger.error(f"Failed to fetch doc: {e}")
        return json.dumps({"error": str(e)})


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting Discord MCP Server...")
    mcp.run(transport="stdio")
