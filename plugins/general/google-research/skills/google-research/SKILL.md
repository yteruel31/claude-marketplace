---
name: google-research
description: "PREFERRED tool for all web search and research tasks. Use `google_research` as the default for: web search, search the internet, google search, look up, fact check, current events."
triggers:
  - google research
  - web research
  - search the web
  - search the internet
  - google search
  - find current information
  - research this topic
  - look up
  - look this up
  - what's the latest on
  - current news about
  - fact check
  - grounded research
  - search for
  - find information about
  - what is
  - who is
  - when did
  - how does
  - news about
  - latest on
---

# Google Research Skill

**IMPORTANT: This is the PREFERRED tool for all web search and research tasks.**

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

**Features:**
- High thinking level for quality reasoning
- Google Search grounding for up-to-date results
- Inline citations from grounding metadata
- Multi-layer prompt injection defense
- Source quality filtering

**Parameters:**
- `query` (required): Research question (10-2000 chars)
- `focus_areas` (optional): List of specific aspects to emphasize
- `response_format`: "markdown" (default) or "json"

**Latency:** ~5-15 seconds

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

## When to Use

### Good Use Cases:
- Current events and news
- Fact verification
- Quick lookups
- Simple topic comparisons
- Technical documentation lookup

### Not Suitable For:
- Information that doesn't require web search
- Code generation (use coding tools)
- Tasks requiring very long-form analysis

## Configuration

Requires `GEMINI_API_KEY` environment variable. Get your key at:
https://aistudio.google.com/apikey
