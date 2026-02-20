---
name: jb-marketplace
description: "JetBrains Marketplace API access. Search plugins, check IDE compatibility, view plugin metadata and version history."
triggers:
  - jetbrains marketplace
  - jetbrains plugin
  - intellij plugin
  - plugin compatibility
  - ide compatibility
  - check plugin compatible
  - plugin versions
  - plugin updates
  - marketplace search
  - search jetbrains plugins
  - plugin metadata
  - plugin info
  - jetbrains marketplace api
  - compatible with ide build
  - plugin version history
---

# JetBrains Marketplace Skill

This skill provides read-only access to the JetBrains Marketplace API.

## Available Tools

### `search_plugins`

Search JetBrains Marketplace plugins by keyword.

**Parameters:**
- `query` (required): Search query
- `build` (optional): IDE build number to filter compatibility (e.g., "IC-251.25410.129")
- `max` (optional): Max results (1-100, default 10)

**Example:**
```
Use search_plugins with query: "git worktree" and build: "IC-251.25410.129"
```

### `get_plugin`

Get detailed metadata for a plugin by its numeric ID.

**Parameters:**
- `plugin_id` (required): Numeric plugin ID from JetBrains Marketplace

**Example:**
```
Use get_plugin with plugin_id: 30140
```

### `list_updates`

List version history and updates for a plugin.

**Parameters:**
- `plugin_id` (required): Numeric plugin ID
- `size` (optional): Number of updates (1-100, default 10)
- `channel` (optional): Release channel filter (e.g., "Stable", "EAP")

**Example:**
```
Use list_updates with plugin_id: 30140 and size: 5
```

### `check_compatible`

Check if plugins are compatible with a specific IDE build number.

**Parameters:**
- `build` (required): IDE build number (e.g., "IC-251.25410.129")
- `plugin_xml_ids` (required): List of plugin XML IDs to check

**Example:**
```
Use check_compatible with build: "IC-251.25410.129" and plugin_xml_ids: ["com.github.yoannteruel.jetbrainsworktreeplugin"]
```

## Common Use Cases

### Check if your plugin works with a specific IDE version
```
Check if com.github.yoannteruel.jetbrainsworktreeplugin is compatible with IC-251.25410.129
```

### Look up plugin download stats
```
Get info for JetBrains Marketplace plugin 30140
```

### View recent releases
```
List the last 5 updates for plugin 30140
```

### Find plugins by keyword
```
Search JetBrains Marketplace for "git worktree"
```

## Workflow: Search then inspect

1. Use `search_plugins` to find a plugin by name
2. Note the numeric `id` from the results
3. Use `get_plugin` with that ID for full metadata
4. Use `list_updates` with that ID for version history

## Limitations

- All endpoints are public and read-only (no authentication needed)
- The search API returns basic fields; use `get_plugin` for full metadata
- `check_compatible` returns empty results if no compatible version exists
- Plugin numeric IDs are different from XML IDs; search returns both
