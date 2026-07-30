# KASKI Nodes

A compact collection of production-oriented custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

KASKI Nodes focuses on the unglamorous but important parts of real workflows: consistent shot and asset IDs, filename-aware loading, reusable API settings, flexible image inputs, string assembly, video conformation, and small orchestration utilities.

> Built for practical AI/VFX pipelines rather than one-off demo workflows.

---

## Highlights

- **IO-unlocked image API nodes** for OpenAI GPT Image and Google Gemini / Nano Banana
- **Central settings nodes** that can feed multiple generators
- **Up to 16 OpenAI reference images** and up to 14 Gemini reference images
- **Production naming tools** for shots, characters, props, locations, and materials
- **Filename-aware image and video loaders**
- **Video frame-count and resolution conformation**
- **Autogrowing string assembly**
- Small workflow helpers such as JSON fragments, number formatting, and asynchronous delay

All nodes are grouped under the `KASKI` category in ComfyUI.

---

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone <repository-url> KASKI-Nodes
```

Restart ComfyUI afterwards.

You can also download the repository as a ZIP and extract it into:

```text
ComfyUI/custom_nodes/KASKI-Nodes
```

### Requirements

KASKI Nodes uses recent ComfyUI APIs, including:

- `comfy_api.latest`
- `comfy_api_nodes`
- dynamic inputs and autogrowing inputs
- ComfyUI's API-node authentication and proxy infrastructure

A current ComfyUI installation is therefore recommended.

The OpenAI and Gemini nodes require API access to be configured through ComfyUI. Local utility nodes do not require external services.

---

## Node Overview

### OpenAI GPT Image

Category: `KASKI/api-adaptions/openai`

#### OpenAI GPT Image Settings

Creates one reusable settings object that can be connected to multiple OpenAI generator nodes.

Supported models:

- `gpt-image-2`
- `gpt-image-1.5`
- `gpt-image-1`

Configurable options include:

- model
- output size
- custom width and height for GPT Image 2
- background mode
- quality
- number of generated images

GPT Image 2 supports additional large and custom output sizes. Custom dimensions must stay within the validation limits enforced by the node.

#### OpenAI GPT Image IO-unlocked

Generates or edits images through the OpenAI image endpoint.

Features:

- text-to-image generation
- image editing with optional reference images
- autogrowing image inputs
- up to **16 individual reference images**
- batched IMAGE tensors
- optional inpainting mask
- multiple outputs per request through the shared settings node
- automatic normalization of returned batch dimensions

A mask requires exactly one reference image. White areas of the mask are replaced.

The `seed` input acts as a ComfyUI cache-buster and supports control-after-generate. It is not sent to the OpenAI API.

---

### Gemini / Nano Banana

Category: `KASKI/api-adaptions/nanobanana`

#### Nanobanana Settings

Creates a reusable settings object for one or more Gemini image generator nodes.

Supported models:

- **Gemini 3 Pro Image**
- **Nano Banana 2 — Gemini 3.1 Flash Image**
- **Nano Banana 2 Lite**

Depending on the selected model, the node exposes:

- aspect ratio
- output resolution
- response modality
- thinking level
- temperature
- top-p
- shared system prompt

The settings output can be fanned out to several generator nodes to keep a workflow consistent.

#### GeminiImage2 IO-unlocked

Generates or edits images through the Google Vertex Gemini endpoint.

Features:

- prompt-based image generation
- single or batched IMAGE reference input
- up to **14 reference images**
- optional compatible Gemini input files
- IMAGE or IMAGE+TEXT responses
- optional thought-image output when supplied by the model
- configurable aspect ratios and resolutions
- central shared settings

Outputs:

1. generated image
2. model text
3. thought image

The `seed` input is used only to invalidate ComfyUI's cache and trigger a new request. It is not sent to Gemini.

---

## ID Tools

Category: `KASKI/ID-Tools`

The ID nodes create and extract predictable production identifiers for shots and reusable references.

### Reference IDs

Reference IDs support:

- optional project name
- reference type
- reference name
- optional artist code
- numeric version or `vN`

Supported reference types:

- `character`
- `prop`
- `location`
- `material`

Format:

```text
[project_]type_name_[artist_]version
```

Examples:

```text
character_father_v1
SOMAT_character_father_v3
SOMAT_character_father_KW_v3
PROJECT_location_kitchen_AB_vN
```

Available nodes:

| Node | Purpose |
|---|---|
| **Generate Reference ID** | Builds a valid reference ID |
| **Extract Reference ID** | Extracts a reference ID from a larger string or filename |

Project names, reference names, and artist codes must not contain spaces or underscores. Hyphens are supported.

### Shot IDs

Shot IDs support:

- optional project name
- zero-padded shot number
- pipeline step
- optional artist code
- numeric version or `vN`

Format:

```text
[project_]sh[number]_pipelineStep_[artist_]version
```

Examples:

```text
sh001_firstFrame_v1
SOMAT_sh012_enhanced_v4
SOMAT_sh012_firstFrame_KW_v5
PROJECT_sh120_cgi_AB_vN
```

Supported pipeline steps:

- `firstFrame`
- `notEnhanced`
- `enhanced`
- `Depth`
- `Normal`
- `Scribble`
- `lastFrame`
- `cgi`

Available nodes:

| Node | Purpose |
|---|---|
| **Generate Shot ID** | Builds a shot ID with configurable zero-padding |
| **Modify Shot ID** | Changes selected parts of an existing valid shot ID |
| **Extract Shot ID** | Extracts a shot ID from a larger string or filename |

`Modify Shot ID` can keep or replace the project, shot number, pipeline step, and artist code. Its version control offers:

- `Keep`
- `Increment`

Numeric versions are incremented normally:

```text
v4 → v5
```

The unresolved version `vN` remains `vN`, including when `Increment` is selected.

---

## String Tools

Category: `KASKI/stringtools`

| Node | Purpose |
|---|---|
| **JSON Key-Value String** | Creates a JSON-style key-value fragment |
| **String Split at Symbol** | Splits a string and returns one indexed segment |
| **Join Strings** | Joins 2–50 dynamically growing string inputs |
| **Number to String** | Formats integers or floats as strings |

### JSON Key-Value String

This node creates a fragment intended for assembling larger JSON-like strings:

```json
"prompt": "cinematic kitchen",
```

It adds a trailing comma and newline. It does **not** validate or return a complete JSON document by itself.

The `nested` option wraps the supplied value in braces when needed.

### Join Strings

The node starts with two string sockets and can grow dynamically up to 50 inputs. Empty disconnected values are ignored.

Typical uses include:

- filename construction
- path assembly
- prompt fragments
- JSON fragments
- metadata strings

### Number to String

Supports:

- integer or float mode
- integer zero-padding
- configurable decimal places

Examples:

```text
7      → 007
3.1416 → 3.14
```

---

## Video Tools

Category: `KASKI/Video`

### Conform Video Size and Length

Constrains an IMAGE sequence by frame count and spatial resolution.

Frame behavior:

- sequences shorter than the minimum are extended with a ping-pong pattern
- sequences longer than the maximum are reduced through evenly distributed frame sampling

Resolution behavior:

- preserves aspect ratio
- scales up to satisfy minimum dimensions
- scales down to satisfy maximum dimensions
- leaves the sequence untouched when it already fits the constraints

Setting a minimum and maximum pair to `0 / 0` preserves the corresponding original property.

### Conform Video for Wan 2.1

Prepares sequence metadata for WAN / VACE-style workflows.

The node:

- extends the input sequence to a valid `4n + 1` frame count
- uses a ping-pong extension pattern
- selects a suitable resolution bucket
- outputs the recommended width and height

Current buckets:

```text
480 × 832
832 × 480
512 × 512
768 × 768
1024 × 1024
1280 × 720
720 × 1280
```

The node does **not** resize the IMAGE sequence itself. It returns the conformed sequence plus the selected target dimensions for downstream resize nodes.

---

## Filename-Aware Loaders

Category: `KASKI/loaders`

### Load Image with Filename

Loads an image while preserving its original filename as a separate STRING output.

Outputs:

1. IMAGE
2. filename
3. MASK

Animated or multi-frame image formats are returned as batches when supported.

### Load Video with Filename

Loads a ComfyUI VIDEO object and returns the source filename alongside it.

Outputs:

1. VIDEO
2. filename

These nodes are useful when downstream naming, metadata, save paths, or ID extraction must remain tied to the source file.

---

## Async Utilities

Category: `KASKI/async`

### Async Delay

Passes an IMAGE through unchanged after a configurable asynchronous delay in milliseconds.

This can be useful for:

- staggering API requests
- testing asynchronous workflow behavior
- introducing simple timing offsets without blocking the entire event loop

---

## Suggested Workflow Patterns

### Centralized API Configuration

Connect one settings node to several generators:

```text
OpenAI GPT Image Settings
    ├── OpenAI GPT Image IO-unlocked
    ├── OpenAI GPT Image IO-unlocked
    └── OpenAI GPT Image IO-unlocked
```

This keeps model, quality, size, and output count synchronized across a workflow.

The same pattern applies to `Nanobanana Settings` and `GeminiImage2 IO-unlocked`.

### Filename-Driven Shot Processing

```text
Load Image with Filename
    ├── IMAGE → processing pipeline
    └── filename → Extract Shot ID → Modify Shot ID → save-name construction
```

### Production Reference Naming

```text
Generate Reference ID
    → PROJECT_character_name_ARTIST_v3
    → metadata / save path / review export
```

---

## Error Behavior

The ID and utility nodes generally raise explicit errors for malformed or contradictory inputs.

The API-adaption nodes use soft error handling:

- the exception and traceback are printed to the ComfyUI console
- the workflow receives a black fallback image instead of terminating immediately
- Gemini text output falls back to an empty string

Check the ComfyUI console when an API node returns black output unexpectedly.

---

## Repository Structure

```text
KASKI-Nodes/
├── __init__.py
├── async_tools.py
├── gptimage_rewrite.py
├── id_tools.py
├── input_conformer.py
├── loaders.py
├── nanobanana_pro_rewrite.py
└── string_tools.py
```

---

## Development Status

These nodes are built around active production needs and recent ComfyUI APIs. API schemas, model identifiers, pricing metadata, and upstream ComfyUI interfaces may change.

When updating ComfyUI, verify the API-adaption nodes before relying on them in unattended or production-critical workflows.

Issues and focused pull requests are welcome.
