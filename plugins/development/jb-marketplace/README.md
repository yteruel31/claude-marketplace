# JetBrains Marketplace MCP Server

MCP server providing read-only access to the [JetBrains Marketplace API](https://plugins.jetbrains.com). Search plugins, check IDE compatibility, view metadata, and browse version history.

## Features

- **Search plugins** by keyword with optional IDE build filter
- **Get plugin metadata** (downloads, ratings, vendor, description, etc.)
- **List version history** with channel filtering
- **Check compatibility** of plugins against specific IDE builds

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

No API key or authentication needed — all endpoints are public.

## Available Tools

| Tool | Description |
|---|---|
| `search_plugins` | Search plugins by keyword |
| `get_plugin` | Get full metadata by numeric plugin ID |
| `list_updates` | List version history for a plugin |
| `check_compatible` | Check if plugins work with a specific IDE build |

## Usage Examples

### Search for plugins
```
Search JetBrains Marketplace for "git worktree"
```

### Check compatibility
```
Check if com.github.yoannteruel.jetbrainsworktreeplugin is compatible with IC-251.25410.129
```

### View version history
```
List the last 5 updates for plugin 30140
```

### Get plugin details
```
Get info for JetBrains Marketplace plugin 30140
```

## Installation

This plugin is part of the [claude-marketplace](https://github.com/yteruel31/claude-marketplace). Install via:

```bash
claude /plugin marketplace add https://github.com/yteruel31/claude-marketplace.git
```

Then enable the `jb-marketplace` plugin.
