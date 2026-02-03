# Google Research Plugin

MCP server for web research using Google Gemini Flash with Google Search grounding.

## Features

- **Real-time web research** with Google Search grounding
- **High thinking level** for quality reasoning
- **Inline citations** from grounding metadata
- **Multi-layer prompt injection defense**
- **Source quality filtering** (deprioritizes SEO-poisoned/spam sites)

## Installation

```bash
/plugin install google-research@claude-marketplace
```

## Requirements

1. **GEMINI_API_KEY** - Get your API key at https://aistudio.google.com/apikey
2. **uv** - Python package manager (https://github.com/astral-sh/uv)

Set the environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Available Tools

### `google_research`

Quick web research with Google Gemini Flash and Google Search grounding.

**Use for:**
- Current events and news
- Fact verification
- Quick lookups
- Topic comparisons
- Technical documentation lookup

**Parameters:**
- `query` (required): Research question (10-2000 chars)
- `focus_areas` (optional): List of specific aspects to emphasize
- `response_format`: "markdown" (default) or "json"

**Example:**
```
Use google_research with query: "What are the latest AI developments in 2025?"
```

## Response Format

Returns:
- **Report**: Research findings with inline citations
- **Sources**: List of URLs with titles
- **Metadata**: Model, execution time, source count

## Security Features

- **Prompt injection prevention**: Content classifiers, security thought reinforcement
- **Source quality filtering**: Deprioritizes SEO-poisoned/spam sites
- **Safety settings**: BLOCK_LOW_AND_ABOVE threshold (most restrictive)

## Configuration

The MCP server is configured in `.mcp.json`:

```json
{
  "mcpServers": {
    "google-research": {
      "command": "bash",
      "args": ["${CLAUDE_PLUGIN_ROOT}/servers/google-research-mcp/run.sh"],
      "env": {
        "GEMINI_API_KEY": "${GEMINI_API_KEY}"
      }
    }
  }
}
```
