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

## Plugin Structure

```
claude-marketplace/
├── .claude/
│   └── settings.json
├── .claude-plugin/
│   └── marketplace.json
├── plugins/
│   └── general/
│       └── google-research/
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
    "google-research@claude-marketplace": true
  }
}
```
