---
name: google-image
version: "1.0.1"
description: "Generate images using AI. Use for: create an image, generate a picture, draw, illustrate, visualize, make an image, text to image, ai art, design a logo, render, sketch."
---

# Google Image Skill

**After calling `generate_image`, ALWAYS use the Read tool on the returned `file_path` to display the image inline to the user.**

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
- `image_size` (optional): Output resolution — `1K` (default, ~1024px), `2K` (~2048px), `4K` (ultra HD)
- `model` (optional): `gemini-2.5-flash-image` (default, fast) or `gemini-3-pro-image-preview` (higher quality, reasoning-driven)

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

### High Resolution
```
Use generate_image with:
  prompt: "Detailed botanical illustration of a rare orchid"
  image_size: "4K"
```

### Custom Output Directory
```
Use generate_image with:
  prompt: "Logo design for a coffee shop"
  output_dir: "assets/images"
```

### Higher Quality Model
```
Use generate_image with:
  prompt: "Photorealistic portrait in oil painting style"
  model: "gemini-3-pro-image-preview"
```

## Response Format

Returns JSON:
- **file_path**: Absolute path to the saved PNG file
- **message**: Instruction to use Read tool for inline display
- **metadata**: Model, aspect ratio, image size, prompt, execution time, file size

## Configuration

Requires `GEMINI_API_KEY` environment variable. Get your key at:
https://aistudio.google.com/apikey
