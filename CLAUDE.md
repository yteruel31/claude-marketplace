# Claude Code Project Memory

## MCP Tool Naming Rules

- The **full namespaced name** must be max **64 characters**: `mcp__plugin_<plugin>_<server>__<tool>`
- Formula: `5 (mcp__) + len(plugin_prefix) + 1 (_) + len(server_name) + 2 (__) + len(tool_name) <= 64`
- Allowed characters: `^[a-zA-Z0-9_-]{1,64}$` (snake_case convention)
- Keep server names short — they consume the character budget
- Avoid redundancy: don't repeat server name inside tool names
