#!/usr/bin/env python3
"""
Reddit Research MCP Server

MCP server providing read-only Reddit research tools using the Reddit API
with OAuth2 client_credentials authentication.

Features:
- Search across Reddit or within specific subreddits
- Browse subreddit feeds (hot/new/top/rising/controversial)
- Read post comments and discussion threads
- Get subreddit info and wiki pages
- Multi-query parallel research with consolidated reports (get_insight)

Usage:
    python reddit_mcp.py
"""

import asyncio
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
from urllib.parse import urlparse

import httpx
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

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API_BASE = "https://oauth.reddit.com"
USER_AGENT = "claude-code:reddit-research-mcp:v1.0.0"

# Token refresh buffer (refresh 5 min before expiry)
TOKEN_REFRESH_BUFFER = 300


# =============================================================================
# Enums
# =============================================================================


class SearchSort(str, Enum):
    RELEVANCE = "relevance"
    HOT = "hot"
    TOP = "top"
    NEW = "new"
    COMMENTS = "comments"


class FeedSort(str, Enum):
    HOT = "hot"
    NEW = "new"
    TOP = "top"
    RISING = "rising"
    CONTROVERSIAL = "controversial"


class CommentSort(str, Enum):
    BEST = "best"
    TOP = "top"
    NEW = "new"
    CONTROVERSIAL = "controversial"
    OLD = "old"
    QA = "qa"


class TimeFilter(str, Enum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    ALL = "all"


class UserSort(str, Enum):
    HOT = "hot"
    NEW = "new"
    TOP = "top"
    CONTROVERSIAL = "controversial"


# =============================================================================
# Input Models (Pydantic)
# =============================================================================


class SearchInput(BaseModel):
    """Input for searching Reddit."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    query: str = Field(
        ...,
        description="Search query",
        min_length=2,
        max_length=512,
    )
    sort: SearchSort = Field(
        default=SearchSort.RELEVANCE,
        description="Sort order: relevance, hot, top, new, comments",
    )
    time_filter: TimeFilter = Field(
        default=TimeFilter.ALL,
        description="Time filter: hour, day, week, month, year, all",
    )
    subreddit: Optional[str] = Field(
        default=None,
        description="Optional subreddit to scope the search to (e.g. 'python')",
    )
    limit: int = Field(
        default=25,
        description="Number of results (1-100)",
        ge=1,
        le=100,
    )


class SubredditPostsInput(BaseModel):
    """Input for browsing subreddit posts."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    subreddit: str = Field(
        ...,
        description="Subreddit name (e.g. 'python', 'machinelearning')",
        min_length=1,
        max_length=100,
    )
    sort: FeedSort = Field(
        default=FeedSort.HOT,
        description="Sort: hot, new, top, rising, controversial",
    )
    time_filter: TimeFilter = Field(
        default=TimeFilter.ALL,
        description="Time filter for top/controversial: hour, day, week, month, year, all",
    )
    limit: int = Field(
        default=25,
        description="Number of posts (1-100)",
        ge=1,
        le=100,
    )


class PostCommentsInput(BaseModel):
    """Input for reading post comments."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    post_id: str = Field(
        ...,
        description=(
            "Post ID (e.g. 'abc123') or full Reddit URL "
            "(e.g. 'https://www.reddit.com/r/python/comments/abc123/...')"
        ),
    )
    subreddit: Optional[str] = Field(
        default=None,
        description="Subreddit name (required if post_id is not a full URL)",
    )
    sort: CommentSort = Field(
        default=CommentSort.BEST,
        description="Comment sort: best, top, new, controversial, old, qa",
    )
    limit: int = Field(
        default=50,
        description="Number of top-level comments (1-100)",
        ge=1,
        le=100,
    )

    @field_validator("post_id")
    @classmethod
    def validate_post_id(cls, v: str) -> str:
        """Strip whitespace."""
        return v.strip()


class SubredditInfoInput(BaseModel):
    """Input for getting subreddit info."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    subreddit: str = Field(
        ...,
        description="Subreddit name (e.g. 'python')",
        min_length=1,
        max_length=100,
    )


class UserOverviewInput(BaseModel):
    """Input for getting user overview."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    username: str = Field(
        ...,
        description="Reddit username (without u/ prefix)",
        min_length=1,
        max_length=100,
    )
    sort: UserSort = Field(
        default=UserSort.NEW,
        description="Sort: hot, new, top, controversial",
    )
    limit: int = Field(
        default=25,
        description="Number of items (1-100)",
        ge=1,
        le=100,
    )


class SubredditWikiInput(BaseModel):
    """Input for reading subreddit wiki pages."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    subreddit: str = Field(
        ...,
        description="Subreddit name (e.g. 'python')",
        min_length=1,
        max_length=100,
    )
    page: str = Field(
        default="index",
        description="Wiki page name (default 'index' lists all pages)",
        max_length=200,
    )


class GetInsightInput(BaseModel):
    """Input for multi-query Reddit research."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    topic: str = Field(
        ...,
        description="Research topic or question",
        min_length=5,
        max_length=500,
    )
    subreddits: Optional[List[str]] = Field(
        default=None,
        description="Specific subreddits to target (max 5)",
        max_length=5,
    )
    include_comments: bool = Field(
        default=True,
        description="Whether to fetch top comments from the most relevant posts",
    )
    max_results: int = Field(
        default=50,
        description="Max total results to return",
        ge=10,
        le=100,
    )

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("Topic must be at least 5 characters")
        return v.strip()


# =============================================================================
# Reddit Client
# =============================================================================


class RedditClient:
    """Handles OAuth2 auth, token refresh, rate limiting, and all Reddit API calls."""

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token: Optional[str] = None
        self.token_expiry: float = 0
        self.http_client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=30.0,
        )
        # Rate limit tracking
        self._rate_remaining: Optional[float] = None
        self._rate_reset: Optional[float] = None

    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()

    async def _authenticate(self):
        """Get or refresh OAuth2 token via client_credentials grant."""
        logger.info("Authenticating with Reddit API (client_credentials)...")

        response = await self.http_client.post(
            TOKEN_URL,
            auth=(self.client_id, self.client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        data = response.json()

        if "access_token" not in data:
            raise ValueError(f"Authentication failed: {data}")

        self.access_token = data["access_token"]
        self.token_expiry = time.time() + data.get("expires_in", 3600)
        logger.info("Reddit API authentication successful")

    async def _ensure_token(self):
        """Ensure we have a valid, non-expired token."""
        if not self.access_token or time.time() >= (self.token_expiry - TOKEN_REFRESH_BUFFER):
            await self._authenticate()

    def _update_rate_limits(self, response: httpx.Response):
        """Parse rate limit headers from response."""
        remaining = response.headers.get("X-Ratelimit-Remaining")
        reset = response.headers.get("X-Ratelimit-Reset")

        if remaining is not None:
            self._rate_remaining = float(remaining)
        if reset is not None:
            self._rate_reset = time.time() + float(reset)

        if self._rate_remaining is not None and self._rate_remaining < 5:
            logger.warning(f"Rate limit low: {self._rate_remaining} remaining")

    async def _wait_for_rate_limit(self):
        """Sleep if we're close to the rate limit."""
        if (
            self._rate_remaining is not None
            and self._rate_remaining < 2
            and self._rate_reset is not None
        ):
            wait_time = self._rate_reset - time.time()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(min(wait_time, 60))

    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated GET request to the Reddit API."""
        await self._ensure_token()
        await self._wait_for_rate_limit()

        url = f"{API_BASE}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": USER_AGENT,
        }

        if params is None:
            params = {}
        params["raw_json"] = 1  # Get unescaped HTML entities

        response = await self.http_client.get(url, headers=headers, params=params)
        self._update_rate_limits(response)

        if response.status_code == 401:
            # Token expired, re-auth and retry once
            await self._authenticate()
            headers["Authorization"] = f"Bearer {self.access_token}"
            response = await self.http_client.get(url, headers=headers, params=params)
            self._update_rate_limits(response)

        response.raise_for_status()
        return response.json()

    # -------------------------------------------------------------------------
    # Data formatting helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _format_post(post_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant fields from a post's data dict."""
        d = post_data.get("data", post_data)
        selftext = d.get("selftext", "") or ""
        return {
            "id": d.get("id"),
            "title": d.get("title"),
            "author": d.get("author"),
            "subreddit": d.get("subreddit"),
            "score": d.get("score"),
            "upvote_ratio": d.get("upvote_ratio"),
            "num_comments": d.get("num_comments"),
            "url": d.get("url"),
            "permalink": f"https://www.reddit.com{d.get('permalink', '')}",
            "selftext": selftext[:500] + ("..." if len(selftext) > 500 else ""),
            "created_utc": d.get("created_utc"),
            "is_self": d.get("is_self"),
            "link_flair_text": d.get("link_flair_text"),
        }

    @staticmethod
    def _format_comment(comment_data: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """Extract relevant fields from a comment's data dict."""
        d = comment_data.get("data", comment_data)
        body = d.get("body", "") or ""
        return {
            "id": d.get("id"),
            "author": d.get("author"),
            "body": body[:1000] + ("..." if len(body) > 1000 else ""),
            "score": d.get("score"),
            "depth": depth,
            "created_utc": d.get("created_utc"),
            "permalink": f"https://www.reddit.com{d.get('permalink', '')}",
        }

    def _flatten_comments(
        self,
        children: List[Dict[str, Any]],
        depth: int = 0,
        max_depth: int = 3,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Recursively flatten comment tree up to max_depth."""
        results = []
        for child in children:
            if len(results) >= limit:
                break
            if child.get("kind") != "t1":
                continue
            results.append(self._format_comment(child, depth))
            # Recurse into replies
            replies = child.get("data", {}).get("replies")
            if replies and isinstance(replies, dict) and depth < max_depth:
                reply_children = replies.get("data", {}).get("children", [])
                results.extend(
                    self._flatten_comments(
                        reply_children,
                        depth=depth + 1,
                        max_depth=max_depth,
                        limit=limit - len(results),
                    )
                )
        return results

    @staticmethod
    def _parse_post_url(url_or_id: str) -> tuple[Optional[str], str]:
        """Parse a Reddit URL or post ID into (subreddit, post_id)."""
        url_or_id = url_or_id.strip()

        # Full URL: https://www.reddit.com/r/python/comments/abc123/...
        if "reddit.com" in url_or_id:
            parsed = urlparse(url_or_id)
            parts = [p for p in parsed.path.split("/") if p]
            # Expected: ['r', 'subreddit', 'comments', 'post_id', ...]
            if len(parts) >= 4 and parts[0] == "r" and parts[2] == "comments":
                return parts[1], parts[3]

        # Just an ID
        return None, url_or_id

    # -------------------------------------------------------------------------
    # API methods
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        sort: str = "relevance",
        time_filter: str = "all",
        subreddit: Optional[str] = None,
        limit: int = 25,
    ) -> Dict[str, Any]:
        """Search Reddit posts."""
        endpoint = f"/r/{subreddit}/search" if subreddit else "/search"
        params: Dict[str, Any] = {
            "q": query,
            "sort": sort,
            "t": time_filter,
            "limit": limit,
            "type": "link",
        }
        if subreddit:
            params["restrict_sr"] = "true"

        data = await self._request(endpoint, params)
        posts = [self._format_post(child) for child in data.get("data", {}).get("children", [])]
        return {
            "query": query,
            "subreddit": subreddit,
            "sort": sort,
            "time_filter": time_filter,
            "count": len(posts),
            "posts": posts,
        }

    async def get_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "hot",
        time_filter: str = "all",
        limit: int = 25,
    ) -> Dict[str, Any]:
        """Get posts from a subreddit feed."""
        endpoint = f"/r/{subreddit}/{sort}"
        params: Dict[str, Any] = {"limit": limit}
        if sort in ("top", "controversial"):
            params["t"] = time_filter

        data = await self._request(endpoint, params)
        posts = [self._format_post(child) for child in data.get("data", {}).get("children", [])]
        return {
            "subreddit": subreddit,
            "sort": sort,
            "count": len(posts),
            "posts": posts,
        }

    async def get_post_comments(
        self,
        subreddit: str,
        post_id: str,
        sort: str = "best",
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get comments on a post."""
        endpoint = f"/r/{subreddit}/comments/{post_id}"
        params: Dict[str, Any] = {"sort": sort, "limit": limit}

        data = await self._request(endpoint, params)

        # Reddit returns [post_listing, comments_listing]
        post_info = {}
        comments = []
        if isinstance(data, list) and len(data) >= 2:
            post_children = data[0].get("data", {}).get("children", [])
            if post_children:
                post_info = self._format_post(post_children[0])

            comment_children = data[1].get("data", {}).get("children", [])
            comments = self._flatten_comments(comment_children, limit=limit)

        return {
            "post": post_info,
            "sort": sort,
            "comment_count": len(comments),
            "comments": comments,
        }

    async def get_subreddit_info(self, subreddit: str) -> Dict[str, Any]:
        """Get subreddit metadata."""
        data = await self._request(f"/r/{subreddit}/about")
        d = data.get("data", data)
        return {
            "name": d.get("display_name"),
            "title": d.get("title"),
            "description": d.get("public_description", "")[:500],
            "long_description": (d.get("description", "") or "")[:1000],
            "subscribers": d.get("subscribers"),
            "active_users": d.get("accounts_active"),
            "created_utc": d.get("created_utc"),
            "over18": d.get("over18"),
            "url": f"https://www.reddit.com/r/{d.get('display_name', subreddit)}/",
        }

    async def get_user_overview(
        self,
        username: str,
        sort: str = "new",
        limit: int = 25,
    ) -> Dict[str, Any]:
        """Get a user's recent posts and comments."""
        endpoint = f"/user/{username}/overview"
        params: Dict[str, Any] = {"sort": sort, "limit": limit}

        data = await self._request(endpoint, params)
        items = []
        for child in data.get("data", {}).get("children", []):
            kind = child.get("kind")
            d = child.get("data", {})
            if kind == "t3":  # Post
                items.append({
                    "type": "post",
                    "subreddit": d.get("subreddit"),
                    "title": d.get("title"),
                    "score": d.get("score"),
                    "num_comments": d.get("num_comments"),
                    "permalink": f"https://www.reddit.com{d.get('permalink', '')}",
                    "created_utc": d.get("created_utc"),
                })
            elif kind == "t1":  # Comment
                body = d.get("body", "") or ""
                items.append({
                    "type": "comment",
                    "subreddit": d.get("subreddit"),
                    "body": body[:500] + ("..." if len(body) > 500 else ""),
                    "score": d.get("score"),
                    "permalink": f"https://www.reddit.com{d.get('permalink', '')}",
                    "created_utc": d.get("created_utc"),
                })

        return {
            "username": username,
            "sort": sort,
            "count": len(items),
            "items": items,
        }

    async def get_subreddit_wiki(
        self,
        subreddit: str,
        page: str = "index",
    ) -> Dict[str, Any]:
        """Get a subreddit wiki page."""
        endpoint = f"/r/{subreddit}/wiki/{page}"
        data = await self._request(endpoint)
        wiki_data = data.get("data", data)
        content = wiki_data.get("content_md", "") or wiki_data.get("content_html", "")
        return {
            "subreddit": subreddit,
            "page": page,
            "content": content[:10000],
            "revision_by": wiki_data.get("revision_by", {}).get("data", {}).get("name"),
            "revision_date": wiki_data.get("revision_date"),
        }

    async def get_insight(
        self,
        topic: str,
        subreddits: Optional[List[str]] = None,
        include_comments: bool = True,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        """
        Multi-query parallel research across Reddit.

        Generates search variations, runs them in parallel, fetches top
        comments, deduplicates, and returns a consolidated report.
        """
        start_time = time.time()

        # Generate search variations
        queries = self._generate_search_variations(topic)

        # Build search tasks: global + per-subreddit
        search_tasks = []
        per_search_limit = max(10, max_results // len(queries))

        for q in queries:
            # Global search
            search_tasks.append(
                self.search(query=q, sort="relevance", time_filter="year", limit=per_search_limit)
            )
            # Per-subreddit searches
            if subreddits:
                for sub in subreddits:
                    search_tasks.append(
                        self.search(
                            query=q,
                            sort="relevance",
                            time_filter="year",
                            subreddit=sub,
                            limit=per_search_limit,
                        )
                    )

        # Run all searches in parallel
        logger.info(f"Running {len(search_tasks)} parallel searches for topic: {topic[:50]}")
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Collect and deduplicate posts
        seen_ids = set()
        all_posts = []
        for result in search_results:
            if isinstance(result, Exception):
                logger.warning(f"Search failed: {result}")
                continue
            for post in result.get("posts", []):
                post_id = post.get("id")
                if post_id and post_id not in seen_ids:
                    seen_ids.add(post_id)
                    all_posts.append(post)

        # Sort by score and take top results
        all_posts.sort(key=lambda p: p.get("score", 0) or 0, reverse=True)
        top_posts = all_posts[:max_results]

        # Fetch comments for top posts if requested
        comments_data = []
        if include_comments and top_posts:
            # Fetch comments for top 5 posts
            comment_posts = top_posts[:5]
            comment_tasks = []
            for post in comment_posts:
                sub = post.get("subreddit")
                pid = post.get("id")
                if sub and pid:
                    comment_tasks.append(
                        self.get_post_comments(
                            subreddit=sub, post_id=pid, sort="top", limit=10
                        )
                    )

            if comment_tasks:
                comment_results = await asyncio.gather(*comment_tasks, return_exceptions=True)
                for result in comment_results:
                    if isinstance(result, Exception):
                        logger.warning(f"Comment fetch failed: {result}")
                        continue
                    comments_data.append(result)

        # Collect unique subreddits
        subreddit_counts: Dict[str, int] = {}
        for post in top_posts:
            sub = post.get("subreddit", "unknown")
            subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1

        execution_time = time.time() - start_time

        report = {
            "topic": topic,
            "metadata": {
                "queries_used": queries,
                "target_subreddits": subreddits,
                "total_posts_found": len(all_posts),
                "unique_posts_returned": len(top_posts),
                "comments_fetched_for": len(comments_data),
                "subreddit_distribution": dict(
                    sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)
                ),
                "execution_time_seconds": round(execution_time, 1),
            },
            "top_posts": top_posts,
            "discussions": comments_data,
        }

        # Save to disk
        self._save_research_to_disk(topic, report)

        return report

    @staticmethod
    def _generate_search_variations(topic: str) -> List[str]:
        """Generate 3-5 search query variations from a topic."""
        queries = [topic]

        # Add a question form if not already a question
        if not topic.rstrip().endswith("?"):
            queries.append(f"{topic}?")

        # Add "best" or "recommend" variant for opinion-seeking
        lower = topic.lower()
        if not any(w in lower for w in ("best", "recommend", "review", "opinion")):
            queries.append(f"best {topic}")

        # Add "experience" variant for personal accounts
        if not any(w in lower for w in ("experience", "story", "stories")):
            queries.append(f"{topic} experience")

        # Add "vs" or "comparison" if topic contains a noun
        if not any(w in lower for w in ("vs", "versus", "compare", "comparison")):
            queries.append(f"{topic} comparison")

        return queries[:5]

    def _save_research_to_disk(self, topic: str, report: Dict[str, Any]) -> Optional[str]:
        """Save get_insight report to docs/reddit-research/ as markdown."""
        try:
            working_dir = os.environ.get("ORIGINAL_WORKING_DIR", os.getcwd())
            research_dir = Path(working_dir) / "docs" / "reddit-research"
            research_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", topic[:50])
            sanitized = re.sub(r"_+", "_", sanitized).strip("_")
            filename = f"{timestamp}_{sanitized}.md"
            filepath = research_dir / filename

            content = self._build_report_markdown(topic, report)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Research saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save research to disk: {e}")
            return None

    @staticmethod
    def _build_report_markdown(topic: str, report: Dict[str, Any]) -> str:
        """Build markdown document from get_insight report."""
        meta = report.get("metadata", {})
        posts = report.get("top_posts", [])
        discussions = report.get("discussions", [])

        lines = [
            "---",
            f"topic: {topic}",
            f"queries: {json.dumps(meta.get('queries_used', []))}",
            f"subreddits_targeted: {json.dumps(meta.get('target_subreddits'))}",
            f"total_posts: {meta.get('total_posts_found', 0)}",
            f"unique_posts: {meta.get('unique_posts_returned', 0)}",
            f"execution_time: {meta.get('execution_time_seconds', 0)}s",
            f"timestamp: {datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "---",
            "",
            f"# Reddit Research: {topic}",
            "",
            "## Summary",
            "",
            f"- **Posts found:** {meta.get('total_posts_found', 0)} total, "
            f"{meta.get('unique_posts_returned', 0)} unique",
            f"- **Discussions analyzed:** {meta.get('comments_fetched_for', 0)} threads",
            f"- **Subreddits:** {', '.join(f'r/{s} ({c})' for s, c in (meta.get('subreddit_distribution', {})).items())}",
            "",
            "## Top Posts",
            "",
        ]

        for i, post in enumerate(posts[:20], 1):
            score = post.get("score", 0)
            comments = post.get("num_comments", 0)
            lines.append(
                f"{i}. **[{post.get('title', 'Untitled')}]({post.get('permalink', '')})**"
            )
            lines.append(
                f"   r/{post.get('subreddit', '?')} · {score} pts · {comments} comments"
            )
            if post.get("selftext"):
                preview = post["selftext"][:200]
                lines.append(f"   > {preview}")
            lines.append("")

        if discussions:
            lines.extend(["## Key Discussions", ""])
            for disc in discussions:
                post_info = disc.get("post", {})
                lines.append(f"### {post_info.get('title', 'Thread')}")
                lines.append(f"r/{post_info.get('subreddit', '?')} · {post_info.get('score', 0)} pts")
                lines.append("")
                for comment in disc.get("comments", [])[:5]:
                    indent = "  " * comment.get("depth", 0)
                    author = comment.get("author", "[deleted]")
                    score = comment.get("score", 0)
                    body = comment.get("body", "")[:300]
                    lines.append(f"{indent}- **u/{author}** ({score} pts): {body}")
                lines.append("")

        return "\n".join(lines)


# =============================================================================
# MCP Server and Lifespan
# =============================================================================

reddit_client: Optional[RedditClient] = None


@asynccontextmanager
async def app_lifespan(app):
    """Initialize resources that persist across requests."""
    global reddit_client

    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        logger.error("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables required")
        raise ValueError(
            "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables required. "
            "Create a script app at https://www.reddit.com/prefs/apps"
        )

    reddit_client = RedditClient(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
    )
    logger.info("Reddit Research MCP Server initialized")

    yield {"client": reddit_client}

    # Cleanup
    await reddit_client.close()
    logger.info("Reddit Research MCP Server shutting down")


# Initialize FastMCP server
mcp = FastMCP("reddit_mcp", lifespan=app_lifespan)


# =============================================================================
# MCP Tools
# =============================================================================


@mcp.tool(
    name="search_reddit",
    annotations={
        "title": "Search Reddit",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def search_reddit(params: SearchInput, ctx: Context) -> str:
    """Search across Reddit or within a specific subreddit.

    Find posts matching a query with sorting and time filtering options.

    Args:
        params (SearchInput): Validated input containing:
            - query (str): Search query (2-512 chars)
            - sort: relevance, hot, top, new, comments
            - time_filter: hour, day, week, month, year, all
            - subreddit (optional): Scope search to a subreddit
            - limit (int): Number of results (1-100)

    Returns:
        str: JSON with matching posts
    """
    try:
        result = await reddit_client.search(
            query=params.query,
            sort=params.sort.value,
            time_filter=params.time_filter.value,
            subreddit=params.subreddit,
            limit=params.limit,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="get_subreddit_posts",
    annotations={
        "title": "Get Subreddit Posts",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def get_subreddit_posts(params: SubredditPostsInput, ctx: Context) -> str:
    """Browse a subreddit's feed (hot, new, top, rising, controversial).

    Args:
        params (SubredditPostsInput): Validated input containing:
            - subreddit (str): Subreddit name
            - sort: hot, new, top, rising, controversial
            - time_filter: For top/controversial — hour, day, week, month, year, all
            - limit (int): Number of posts (1-100)

    Returns:
        str: JSON with subreddit posts
    """
    try:
        result = await reddit_client.get_subreddit_posts(
            subreddit=params.subreddit,
            sort=params.sort.value,
            time_filter=params.time_filter.value,
            limit=params.limit,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Get subreddit posts failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="get_post_comments",
    annotations={
        "title": "Get Post Comments",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def get_post_comments(params: PostCommentsInput, ctx: Context) -> str:
    """Read a post's discussion thread with comments.

    Accepts a full Reddit URL or a post ID + subreddit name.

    Args:
        params (PostCommentsInput): Validated input containing:
            - post_id (str): Post ID or full Reddit URL
            - subreddit (optional): Required if post_id is not a URL
            - sort: best, top, new, controversial, old, qa
            - limit (int): Number of comments (1-100)

    Returns:
        str: JSON with post details and flattened comment tree
    """
    try:
        # Parse URL or use raw ID
        parsed_sub, parsed_id = RedditClient._parse_post_url(params.post_id)
        subreddit = parsed_sub or params.subreddit
        post_id = parsed_id

        if not subreddit:
            return json.dumps({
                "error": "subreddit is required when post_id is not a full Reddit URL"
            })

        result = await reddit_client.get_post_comments(
            subreddit=subreddit,
            post_id=post_id,
            sort=params.sort.value,
            limit=params.limit,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Get post comments failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="get_subreddit_info",
    annotations={
        "title": "Get Subreddit Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def get_subreddit_info(params: SubredditInfoInput, ctx: Context) -> str:
    """Get metadata about a subreddit — subscribers, description, rules.

    Useful for discovering if a subreddit is relevant to a research topic.

    Args:
        params (SubredditInfoInput): Validated input containing:
            - subreddit (str): Subreddit name

    Returns:
        str: JSON with subreddit metadata
    """
    try:
        result = await reddit_client.get_subreddit_info(subreddit=params.subreddit)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Get subreddit info failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="get_user_overview",
    annotations={
        "title": "Get User Overview",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def get_user_overview(params: UserOverviewInput, ctx: Context) -> str:
    """Get a Reddit user's recent posts and comments.

    Args:
        params (UserOverviewInput): Validated input containing:
            - username (str): Reddit username (without u/ prefix)
            - sort: hot, new, top, controversial
            - limit (int): Number of items (1-100)

    Returns:
        str: JSON with user's recent activity
    """
    try:
        result = await reddit_client.get_user_overview(
            username=params.username,
            sort=params.sort.value,
            limit=params.limit,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Get user overview failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="get_subreddit_wiki",
    annotations={
        "title": "Get Subreddit Wiki",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def get_subreddit_wiki(params: SubredditWikiInput, ctx: Context) -> str:
    """Access a subreddit's wiki pages.

    Many subreddits have comprehensive knowledge bases in their wikis
    (e.g., r/personalfinance, r/fitness, r/programming).

    Args:
        params (SubredditWikiInput): Validated input containing:
            - subreddit (str): Subreddit name
            - page (str): Wiki page name (default 'index' lists all pages)

    Returns:
        str: JSON with wiki page content
    """
    try:
        result = await reddit_client.get_subreddit_wiki(
            subreddit=params.subreddit,
            page=params.page,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Get subreddit wiki failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="get_insight",
    annotations={
        "title": "Reddit Deep Research",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def get_insight(params: GetInsightInput, ctx: Context) -> str:
    """Multi-query Reddit research that runs parallel searches and returns a consolidated report.

    Generates search variations, searches globally and in targeted subreddits,
    fetches top comments from the most relevant posts, deduplicates results,
    and saves a structured report to docs/reddit-research/.

    Args:
        params (GetInsightInput): Validated input containing:
            - topic (str): Research topic or question (5-500 chars)
            - subreddits (optional): Specific subreddits to target (max 5)
            - include_comments (bool): Fetch top comments (default true)
            - max_results (int): Max results (10-100, default 50)

    Returns:
        str: Structured markdown report with top posts, discussions, and metadata
    """
    try:
        logger.info(f"Starting Reddit insight research: {params.topic[:50]}...")

        result = await reddit_client.get_insight(
            topic=params.topic,
            subreddits=params.subreddits,
            include_comments=params.include_comments,
            max_results=params.max_results,
        )

        meta = result["metadata"]
        logger.info(
            f"Research completed: {meta['unique_posts_returned']} posts, "
            f"{meta['comments_fetched_for']} discussions, "
            f"{meta['execution_time_seconds']}s"
        )

        # Return formatted markdown report
        return RedditClient._build_report_markdown(params.topic, result)

    except Exception as e:
        logger.error(f"Reddit insight research failed: {e}")
        return json.dumps({"error": str(e), "help": "Try a different topic or check credentials."})


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting Reddit Research MCP Server...")
    mcp.run(transport="stdio")
