# Tungsten Provider

[Tungsten.run](https://tungsten.run) is an AI image generator supporting Stable Diffusion, Flux, and other models with LoRA/embedding support.

## Endpoint

```
POST /v1/tungsten/generations
```

## Pricing

Fixed **$0.01 per generation** (all models, all sizes).

## Request Format

```json
{
  "type": "z_image_turbo",
  "data": {
    "model_version_uuid": "MODEL_UUID",
    "prompt": "your prompt here",
    "negative_prompt": "things to avoid",
    "num_images": 1,
    "sampler": "euler",
    "steps": 10,
    "cfg": 1.0,
    "clip_skip": 2,
    "width": 1024,
    "height": 1440,
    "loras": [],
    "embeddings": [],
    "controlnets": [],
    "img2img": null
  }
}
```

### Types

| Type | Description | Recommended Steps | Recommended CFG |
|---|---|---|---|
| `sdxl` | Stable Diffusion XL (Illustrious, Pony, photorealistic) | 25–35 | 5–7 |
| `z_image_turbo` | Fast generation (Z-Image Turbo) | 10 | 1.0 |
| `z_image_base` | Z-Image base pipeline | varies | varies |
| `flux1d` | Flux.1 Dev models | 20+ | 3.5 |
| `flux_chroma` | Flux Chroma | varies | varies |
| `flux_kontext` | Flux Kontext | varies | varies |
| `flux_fill` | Flux Fill (inpainting) | varies | varies |
| `qwen_image` | Qwen Image (Alibaba) | varies | varies |
| `sd` | Stable Diffusion 1.5 | 20–30 | 7 |
| `sd35` | Stable Diffusion 3.5 | varies | varies |
| `upscale` | Image upscaling | — | — |
| `face_detailer` | Face detail enhancement | — | — |

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `model_version_uuid` | string | Model ID from Tungsten (see below) |
| `prompt` | string | Text prompt for generation |
| `negative_prompt` | string | Things to avoid in the output |
| `num_images` | int | Number of images to generate (1–4) |
| `sampler` | string | Sampling algorithm (see list below) |
| `steps` | int | Denoising steps (more = higher quality, slower) |
| `cfg` | float | Classifier-free guidance scale |
| `clip_skip` | int | CLIP layers to skip (usually 2) |
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `loras` | array | LoRA models (see below) |
| `embeddings` | array | Textual inversions (see below) |
| `seed` | int | Optional, for reproducible results |

### Available Samplers

`euler`, `euler_ancestral`, `dpmpp_2m_karras`, `dpmpp_sde_karras`, `lms`, `heun`, `dpm_2`, `ddim`, `lcm`

### LoRA Format

```json
{
  "loras": [
    {"model_version_uuid": "LORA_UUID", "weight": 0.5},
    {"model_version_uuid": "LORA_UUID_2", "weight": 0.35}
  ]
}
```

### Embedding Format

```json
{
  "embeddings": [
    {"model_version_uuid": "EMBEDDING_UUID", "weight": 0.75}
  ]
}
```

## Finding Models and LoRAs

Browse available models: **[tungsten.run/model_feed](https://tungsten.run/model_feed)**

Each model page shows:
- Model UUID in the URL (e.g. `8bhodKtg2B`) — this is **not** the API field
- **Version UUID** (e.g. `5xsQEXnZn6` for ZIT v2.0) — use this as `model_version_uuid`
- Find it in page source (`modelVersionUUID` in `__NEXT_DATA__`) or DevTools → Network when clicking **Run**
- Compatible LoRAs and embeddings
- Recommended generation parameters

## Response Format

Images are returned as base64-encoded PNG:

```json
{
  "data": {
    "images": [
      {
        "base64_png": "iVBORw0KGgo..."
      }
    ],
    "count": 1
  }
}
```

## Example: Z-Image Turbo (fast photorealistic)

```bash
curl -X POST https://x402gate.io/v1/tungsten/generations \
  -H "Content-Type: application/json" \
  -d '{
    "type": "z_image_turbo",
    "data": {
      "model_version_uuid": "5xsQEXnZn6",
      "prompt": "beautiful portrait, photorealistic, detailed skin, soft natural lighting, masterpiece",
      "negative_prompt": "worst_quality, bad_quality, lowres, deformed, blurry",
      "num_images": 1,
      "sampler": "euler",
      "steps": 8,
      "cfg": 1.0,
      "clip_skip": 2,
      "width": 1024,
      "height": 1440,
      "loras": [],
      "embeddings": [],
      "controlnets": [],
      "img2img": null
    }
  }'
```

## Example: Anime SDXL with LoRAs

```bash
curl -X POST https://x402gate.io/v1/tungsten/generations \
  -H "Content-Type: application/json" \
  -d '{
    "type": "sdxl",
    "data": {
      "model_version_uuid": "BJN9tyLbsY",
      "prompt": "1girl, anime style, detailed eyes, masterpiece",
      "negative_prompt": "bad quality,worst quality,sketch,censor",
      "num_images": 1,
      "sampler": "euler_ancestral",
      "steps": 30,
      "cfg": 5.5,
      "clip_skip": 2,
      "width": 832,
      "height": 1216,
      "loras": [
        {"model_version_uuid": "4ZXZZoqd5x", "weight": 0.5},
        {"model_version_uuid": "Jv8b5eApyr", "weight": 0.35}
      ],
      "embeddings": [
        {"model_version_uuid": "5AUTocEdT9", "weight": 0.75},
        {"model_version_uuid": "Kdt9kaTwJP", "weight": 0.75}
      ],
      "controlnets": [],
      "img2img": null
    }
  }'
```
