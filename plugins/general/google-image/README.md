# Google Image Plugin

MCP server for AI image generation using Google Gemini.

## Features

- **Text-to-image generation** using Gemini 2.0 Flash
- **Configurable output directory** — save images wherever you need
- **Automatic file saving** with timestamped filenames
- **Inline display** in Claude Code via the Read tool
- **Configurable aspect ratio** and model selection

## Installation

```bash
/plugin install google-image@claude-marketplace
```

## Requirements

1. **GEMINI_API_KEY** — Get your API key at https://aistudio.google.com/apikey
2. **uv** — Python package manager (https://github.com/astral-sh/uv)

Set the environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Available Tools

### `generate_image`

Generate an image from a text prompt using Google Gemini.

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `prompt` | Yes | — | Image description (5-2000 chars) |
| `output_dir` | No | `generated-images/` | Output directory (relative or absolute) |
| `aspect_ratio` | No | `1:1` | `1:1`, `16:9`, `9:16`, `4:3`, `3:4` |
| `model` | No | `gemini-2.0-flash-preview-image-generation` | Gemini model to use |

**Available models:**

- `gemini-2.0-flash-preview-image-generation` — Higher quality (default)
- `gemini-2.0-flash-lite-preview-image-generation` — Faster, lighter

**Example:**

```
Generate an image of a serene mountain lake at sunset with reflections
```

## Response

Returns JSON with:

- **file_path**: Absolute path to the saved PNG image
- **message**: Instruction to use the Read tool to display inline
- **metadata**: Model, aspect ratio, prompt, execution time, file size
