# Claude Marketplace

Personal Claude Code plugins marketplace.

## Quick Start

### Install the Marketplace

```bash
/plugin marketplace add https://github.com/yteruel31/claude-marketplace.git
```

### Browse Available Plugins

```bash
/plugin
```

### Install a Plugin

```bash
/plugin install google-research@claude-marketplace
```

## Available Plugins

| Plugin | Version | Tools | Description |
|--------|---------|-------|-------------|
| `google-research` | 1.0.0 | 1 | Web research with Google Gemini Flash + Search grounding |
| `google-image` | 1.0.0 | 1 | AI image generation with Google Gemini |
| `discord` | 1.1.1 | 10 | Discord bot integration and API documentation access |
| `reddit` | 1.0.0 | 7 | Read-only Reddit research tools |
| `jb-marketplace` | 1.0.0 | 4 | JetBrains Marketplace API (search, compatibility, metadata, versions) |

### google-research

MCP server for web research using Google Gemini Flash with Google Search grounding.

**Features:**
- Real-time web research with Google Search grounding
- High thinking level for quality reasoning
- Inline citations from grounding metadata
- Multi-layer prompt injection defense

**Requirements:**
- `GEMINI_API_KEY` environment variable (get one at https://aistudio.google.com/apikey)
- `uv` package manager (https://github.com/astral-sh/uv)

### google-image

MCP server for AI image generation using Google Gemini.

**Features:**
- Text-to-image generation via Gemini API
- Configurable output directory
- Automatic file saving to disk with timestamped filenames
- Inline display in Claude Code via Read tool
- Configurable aspect ratio and model selection

**Requirements:**
- `GEMINI_API_KEY` environment variable (get one at https://aistudio.google.com/apikey)
- `uv` package manager (https://github.com/astral-sh/uv)

### discord

MCP server for Discord bot integration and API documentation access.

**Features:**
- Send messages, read history, and search channels
- List and inspect servers and channels
- Browse and search Discord API documentation (no auth required)

**Requirements:**
- `DISCORD_BOT_TOKEN` environment variable (from [Discord Developer Portal](https://discord.com/developers/applications))
- `uv` package manager (https://github.com/astral-sh/uv)

### reddit

MCP server providing read-only Reddit research tools.

**Features:**
- Search across Reddit or within specific subreddits
- Browse subreddit feeds, comments, and wiki pages
- Multi-query parallel research with consolidated reports (`get_insight`)

**Requirements:**
- `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` environment variables (from [Reddit App Preferences](https://www.reddit.com/prefs/apps))
- `uv` package manager (https://github.com/astral-sh/uv)

### jb-marketplace

MCP server for the JetBrains Marketplace API.

**Features:**
- Search plugins by keyword with optional IDE build filter
- Check plugin compatibility against specific IDE builds
- Get full plugin metadata (downloads, ratings, vendor, tags)
- Browse version history with channel filtering

**Requirements:**
- `uv` package manager (https://github.com/astral-sh/uv)
- No API key needed — all endpoints are public

## Plugin Structure

```
claude-marketplace/
├── .claude/
│   └── settings.json
├── .claude-plugin/
│   └── marketplace.json
├── plugins/
│   ├── general/
│   │   ├── discord/
│   │   ├── google-image/
│   │   ├── google-research/
│   │   └── reddit/
│   └── development/
│       └── jb-marketplace/
├── LICENSE
└── README.md
```

## Configuration

Add to your project's `.claude/settings.json` for automatic installation:

```json
{
  "extraKnownMarketplaces": {
    "claude-marketplace": {
      "source": {
        "source": "github",
        "repo": "yteruel31/claude-marketplace"
      }
    }
  },
  "enabledPlugins": {
    "google-research@claude-marketplace": true,
    "google-image@claude-marketplace": true,
    "discord@claude-marketplace": true,
    "reddit@claude-marketplace": true,
    "jb-marketplace@claude-marketplace": true
  }
}
```

## License

MIT
