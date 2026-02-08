---
name: reddit
description: "Reddit research tools. Search posts, read discussions, browse subreddits, access wikis, and run multi-query research with consolidated reports."
triggers:
  - reddit
  - search reddit
  - reddit research
  - reddit posts
  - subreddit
  - what does reddit say
  - reddit opinions
  - reddit discussion
  - browse reddit
  - reddit comments
  - reddit wiki
  - reddit insight
  - reddit community
  - ask reddit
  - reddit thread
---

# Reddit Research Skill

This skill provides read-only Reddit research tools for searching, browsing, and analyzing Reddit content.

## Available Tools

### `search_reddit`

Search across Reddit or within a specific subreddit.

**Parameters:**
- `query` (required): Search query (2-512 chars)
- `sort` (optional): relevance, hot, top, new, comments (default: relevance)
- `time_filter` (optional): hour, day, week, month, year, all (default: all)
- `subreddit` (optional): Scope search to a subreddit
- `limit` (optional): Number of results, 1-100 (default: 25)

**Example:**
```
Use search_reddit with query: "best python web framework 2026" and sort: "top" and time_filter: "year"
```

### `get_subreddit_posts`

Browse a subreddit's feed.

**Parameters:**
- `subreddit` (required): Subreddit name (e.g. "python")
- `sort` (optional): hot, new, top, rising, controversial (default: hot)
- `time_filter` (optional): For top/controversial — hour, day, week, month, year, all
- `limit` (optional): Number of posts, 1-100 (default: 25)

**Example:**
```
Use get_subreddit_posts with subreddit: "machinelearning" and sort: "top" and time_filter: "week"
```

### `get_post_comments`

Read a post's discussion thread with comments.

**Parameters:**
- `post_id` (required): Post ID or full Reddit URL
- `subreddit` (optional): Required if post_id is not a full URL
- `sort` (optional): best, top, new, controversial, old, qa (default: best)
- `limit` (optional): Number of comments, 1-100 (default: 50)

**Example:**
```
Use get_post_comments with post_id: "https://www.reddit.com/r/python/comments/abc123/some_post/"
```

### `get_subreddit_info`

Get metadata about a subreddit — subscribers, description, activity.

**Parameters:**
- `subreddit` (required): Subreddit name

**Example:**
```
Use get_subreddit_info with subreddit: "LocalLLaMA"
```

### `get_user_overview`

Get a user's recent posts and comments.

**Parameters:**
- `username` (required): Reddit username (without u/ prefix)
- `sort` (optional): hot, new, top, controversial (default: new)
- `limit` (optional): Number of items, 1-100 (default: 25)

**Example:**
```
Use get_user_overview with username: "spez" and limit: 10
```

### `get_subreddit_wiki`

Access subreddit wiki pages — many subreddits have comprehensive knowledge bases.

**Parameters:**
- `subreddit` (required): Subreddit name
- `page` (optional): Wiki page name (default "index" lists all pages)

**Example:**
```
Use get_subreddit_wiki with subreddit: "personalfinance" and page: "index"
```

### `get_insight`

Multi-query Reddit research. Runs parallel searches, fetches top comments, deduplicates, and saves a structured report to `docs/reddit-research/`.

**Parameters:**
- `topic` (required): Research topic or question (5-500 chars)
- `subreddits` (optional): Specific subreddits to target (max 5)
- `include_comments` (optional): Fetch top comments (default: true)
- `max_results` (optional): Max results, 10-100 (default: 50)

**Example:**
```
Use get_insight with topic: "Is Rust replacing C++ in production systems?" and subreddits: ["rust", "cpp", "programming"]
```

## Common Use Cases

### Research a topic
```
Search Reddit for discussions about "microservices vs monolith in 2026"
```

### Get community opinions
```
What does Reddit say about the new MacBook Pro?
```

### Deep research with report
```
Use get_insight to research "best practices for LLM fine-tuning" across r/LocalLLaMA and r/MachineLearning
```

### Explore a community
```
Get info about the r/ExperiencedDevs subreddit and browse their top posts this month
```

### Read a discussion thread
```
Read the comments on this Reddit post: https://www.reddit.com/r/programming/comments/abc123/
```

## Configuration

Requires `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` environment variables.

**Setup:**
1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app..."
3. Select "script" type
4. Set redirect URI to `http://localhost:8080` (not used but required)
5. Copy the Client ID (under the app name) and Client Secret

```bash
export REDDIT_CLIENT_ID="your-client-id"
export REDDIT_CLIENT_SECRET="your-client-secret"
```

## Limitations

- Read-only: cannot post, vote, or modify content
- Public data only: no access to private subreddits or user inboxes
- Search results capped at 1,000 most recent
- Rate limit: 60 requests/min (handled automatically)
- NSFW content may be blocked for application-only auth
- Wiki pages may not be available on all subreddits
