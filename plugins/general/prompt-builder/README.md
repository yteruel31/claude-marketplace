# Prompt Builder

Generate optimized, paste-ready prompts for AI tools. Supports **Claude.ai**, **Claude Code**, **Gemini**, **Google Stitch**, and **Perplexity**.

## What It Does

You describe what you want. Prompt Builder identifies the target tool, extracts your intent, selects the right framework, applies provider-specific optimizations, and delivers a single copyable prompt ready to paste.

**Key features:**
- Provider-specific optimization for 5 AI tools
- Automatic framework selection (RTF, CO-STAR, RISEN, CoT, Few-Shot, Agentic)
- Anti-pattern detection — silently fixes 20 common prompt failures
- Memory blocks for multi-session context carryover
- Token-efficient output — every word is load-bearing

## Supported Providers

| Provider | Key Optimizations |
|----------|-------------------|
| **Claude.ai** | XML tags, extended thinking, prefilling, long-context positioning |
| **Claude Code** | Agentic contracts, stop conditions, file scoping, phase-gating |
| **Gemini** | Boundary anchoring, grounding, format locks, multimodal referencing |
| **Google Stitch** | DESIGN.md integration, PTCF framework, incremental iteration, bloat prevention |
| **Perplexity** | GIC briefings, mode selection, focus modes, citation control |

## Installation

### Via Claude Marketplace
```bash
/plugin install prompt-builder@claude-marketplace
```

### Manual
1. Clone this repository
2. Copy the `plugins/general/prompt-builder/skills/prompt-builder/` folder
3. Place it in `~/.claude/skills/prompt-builder/`

## Usage Examples

**Simple:**
> "Write a prompt for Gemini to analyze a quarterly earnings report"

**Improving existing prompts:**
> "Fix this prompt: 'help me with my code'"

**Claude Code agentic:**
> "Build a Claude Code prompt to refactor the auth module"

**Google Stitch design:**
> "Write a Stitch prompt for a fintech dashboard"

**Perplexity research:**
> "Create a Perplexity prompt to compare React vs Vue for enterprise"

## How It Works

1. **Detects target tool** from your request
2. **Extracts intent** across 9 dimensions (task, format, constraints, etc.)
3. **Asks max 3 questions** if critical info is missing
4. **Routes to the right framework** silently (you never see framework names)
5. **Applies provider-specific techniques** (XML for Claude, grounding for Gemini, etc.)
6. **Scans for anti-patterns** and fixes them
7. **Delivers one clean prompt** ready to paste

## Inspired By

[prompt-master](https://github.com/nidhinjs/prompt-master) by nidhinjs — adapted and focused for a specific 5-tool workflow.

## License

MIT
