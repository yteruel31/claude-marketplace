# React Flow Documentation Plugin

MCP server for browsing, searching, and retrieving [React Flow](https://reactflow.dev/) documentation. Enables Claude to look up API references, learn guides, and UI component docs for building node-based UIs with `@xyflow/react`.

## Features

- **Browse** all available documentation pages by category (API Reference, Learn, UI)
- **Inspect** page metadata and section structure before fetching content
- **Retrieve** full page content or specific sections
- **Search** across titles, descriptions, and page names

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

No API keys or environment variables required — documentation is bundled with the plugin.

## Tools

| Tool | Description |
|------|-------------|
| `list_pages` | List available documentation pages with optional category filter |
| `get_page_info` | Get page metadata and available sections |
| `get_page` | Get full markdown content or a specific section |
| `search_docs` | Search documentation by keyword |

## Usage Examples

```
# List all API reference pages
list_pages(category="api-reference")

# Get info about the ReactFlow component page
get_page_info(page_name="api-reference/react-flow")

# Read the Props section
get_page(page_name="api-reference/react-flow", section_name="Props")

# Search for custom node documentation
search_docs(query="custom node")
```

## Refreshing Documentation

The bundled `docs.json` is a point-in-time snapshot. To fetch the latest documentation from the [xyflow/web](https://github.com/xyflow/web) repository:

```bash
cd servers/react-flow-mcp
uv run python fetch_docs.py
```

This fetches documentation from GitHub and writes `data/docs.json`.

## License

MIT
