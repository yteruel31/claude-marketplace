---
name: react-flow
version: "1.0.1"
description: "React Flow documentation tools. Browse, search, and retrieve @xyflow/react docs for node-based UIs, flow editors, custom nodes, custom edges, interactive diagrams. Also covers: reactflow, xyflow, react flow hooks, react flow api."
---

# React Flow Documentation Skill

This skill provides read-only tools for browsing, searching, and retrieving React Flow (@xyflow/react) documentation.

## Available Tools

### `list_pages`

List available documentation pages with optional category filtering.

**Parameters:**
- `include_description` (optional): Include page descriptions in output (default: false)
- `category` (optional): Filter by category — api-reference, learn, or ui

**Example:**
```
Use list_pages with category: "api-reference" to see all API reference pages
```

### `get_page_info`

Get page metadata and list of available sections.

**Parameters:**
- `page_name` (required): Page name (e.g., "api-reference/react-flow", "learn/concepts/building-a-flow")

**Example:**
```
Use get_page_info with page_name: "api-reference/react-flow" to see what sections are available
```

### `get_page`

Get the full markdown content for a page, or a specific section.

**Parameters:**
- `page_name` (required): Page name
- `section_name` (optional): Section name to fetch (use get_page_info to see available sections)

**Example:**
```
Use get_page with page_name: "api-reference/react-flow" and section_name: "Props"
```

### `search_docs`

Search documentation by title, description, or page name.

**Parameters:**
- `query` (required): Search query

**Example:**
```
Use search_docs with query: "custom node" to find relevant documentation
```

## Recommended Workflow

For most documentation lookups, chain the tools in this order:

1. **Discover** — `list_pages` with a category filter to find relevant page names
2. **Inspect** — `get_page_info` on a page to see its available sections
3. **Retrieve** — `get_page` with a specific section to get the content you need

For targeted queries, use `search_docs` first to find the right page, then `get_page` to retrieve it.

## Common Use Cases

- **Look up a component API**: "What props does the ReactFlow component accept?"
- **Learn about custom nodes**: "How do I create custom nodes in React Flow?"
- **Find documentation on hooks**: "Search the React Flow docs for useReactFlow hook"
- **Browse available tutorials**: "List all React Flow learn pages"
- **Get edge customization details**: "Show me the React Flow docs on custom edges"

## Limitations

- Read-only: provides documentation only, does not modify any files
- Documentation is a point-in-time snapshot — run `fetch_docs.py` to refresh
- Search matches on title, description, and page name (not full-text content search)
