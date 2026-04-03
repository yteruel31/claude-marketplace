---
name: google-research
version: "1.0.1"
description: "Preferred tool for web search and research. Use for: google search, search the internet, look up, fact check, current events, latest news, what is, who is, research a topic, find information about."
---

# Google Research Skill

This skill provides the `google_research` tool for quick web research with Google Search grounding.

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

**Latency:** ~5-15 seconds

## Operational Guidance

- **Query formulation**: Write natural questions, not keyword strings. "What are the latest React 19 features?" works better than "React 19 features list."
- **When to use `focus_areas`**: Use when the topic is broad and you want to narrow the response. Provide 2-4 specific aspects (e.g., `["performance", "ecosystem", "learning curve"]`).
- **Result presentation**: Summarize the key findings for the user. Include source URLs when citing specific claims. If the results are thin or inconclusive, say so rather than padding.

## Examples

### Basic Research
```
Use google_research with query: "What are the top AI developments in January 2025?"
```

### Focused Research
```
Use google_research with:
  query: "Compare React vs Vue for enterprise applications"
  focus_areas: ["performance", "ecosystem", "learning curve"]
```

### Fact Checking
```
Use google_research with query: "Is it true that [claim to verify]?"
```

### Technical Lookup
```
Use google_research with query: "How to implement authentication in Next.js 15"
```

## Response Format

Returns:
- **Report**: Research findings with inline citations
- **Sources**: List of URLs with titles
- **Metadata**: Model, execution time, source count

## Security Features

Google's multi-layer defense:
- **Prompt injection prevention**: Content classifiers, security thought reinforcement
- **Source quality filtering**: Deprioritizes SEO-poisoned/spam sites
- **Safety settings**: BLOCK_LOW_AND_ABOVE threshold (most restrictive)

### Not Suitable For:
- Information that doesn't require web search
- Code generation (use coding tools)
- Tasks requiring very long-form analysis

## Configuration

Requires `GEMINI_API_KEY` environment variable. Get your key at:
https://aistudio.google.com/apikey
