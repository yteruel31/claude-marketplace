# Reddit Research Plugin

MCP server providing read-only Reddit research tools for Claude Code.

## Features

- **Search** across Reddit or within specific subreddits
- **Browse** subreddit feeds (hot/new/top/rising/controversial)
- **Read** post comments and discussion threads
- **Discover** subreddit info and wiki pages
- **Research** with `get_insight` — parallel multi-query research with saved reports

## Setup

1. Create a Reddit app at https://www.reddit.com/prefs/apps (select "script" type)
2. Set environment variables:

```bash
export REDDIT_CLIENT_ID="your-client-id"
export REDDIT_CLIENT_SECRET="your-client-secret"
```

## Tools

| Tool | Description |
|------|-------------|
| `search_reddit` | Search posts across Reddit |
| `get_subreddit_posts` | Browse a subreddit's feed |
| `get_post_comments` | Read a post's discussion thread |
| `get_subreddit_info` | Get subreddit metadata |
| `get_user_overview` | Get a user's recent activity |
| `get_subreddit_wiki` | Access subreddit wiki pages |
| `get_insight` | Multi-query parallel research with reports |

## Authentication

Uses OAuth2 `client_credentials` grant — only needs Client ID and Secret (no username/password). Tokens auto-refresh every hour. Rate limit: 60 req/min (handled automatically).
