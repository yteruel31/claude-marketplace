---
name: google-image
description: "Generate images using AI. Use `generate_image` for: create an image, generate a picture, draw, illustrate, visualize, make an image."
triggers:
  - generate image
  - create image
  - make image
  - draw
  - illustrate
  - visualize
  - picture of
  - image of
  - generate a picture
  - create a picture
  - text to image
  - ai image
  - ai art
---

# Google Image Skill

**IMPORTANT: After calling `generate_image`, ALWAYS use the Read tool on the returned `file_path` to display the image inline to the user.**

This skill provides the `generate_image` tool for AI image generation with Google Gemini.

## Available Tools

### `generate_image`

Generate an image from a text prompt using Google Gemini.

**Use for:**
- Creating illustrations or concept art
- Generating visual assets
- Visualizing ideas or concepts
- Creating placeholder images

**Parameters:**
- `prompt` (required): Image description (5-2000 chars)
- `output_dir` (optional): Custom output directory (default: `generated-images/`)
- `aspect_ratio` (optional): `1:1` (default), `16:9`, `9:16`, `4:3`, `3:4`
- `model` (optional): Gemini model to use

**Latency:** ~5-20 seconds

## Workflow

1. Call `generate_image` with the prompt
2. Parse the JSON response to get `file_path`
3. **ALWAYS** use the Read tool on `file_path` to display the image inline

## Examples

### Basic Image
```
Use generate_image with prompt: "A serene mountain lake at sunset with reflections"
```

### Wide Format
```
Use generate_image with:
  prompt: "Panoramic view of a futuristic city skyline"
  aspect_ratio: "16:9"
```

### Custom Output Directory
```
Use generate_image with:
  prompt: "Logo design for a coffee shop"
  output_dir: "assets/images"
```

## Response Format

Returns JSON:
- **file_path**: Absolute path to the saved PNG file
- **message**: Instruction to use Read tool for inline display
- **metadata**: Model, aspect ratio, prompt, execution time, file size

## Configuration

Requires `GEMINI_API_KEY` environment variable. Get your key at:
https://aistudio.google.com/apikey
