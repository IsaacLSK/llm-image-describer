# llm-image-testing

Image description pipeline using xAI Grok vision models.

This project reads car images, sends them to Grok, and stores structured description results in JSON.

## What This Project Does

- Reads images from a directory (`IMAGE_INPUT`)
- Describes each image with Grok vision
- Writes results to a JSON file (`IMAGE_OUTPUT`)
- Supports server-style route calls for:
	- single image by name
	- single image by file path
	- batch by directory path

## Project Files

- `grok_image_describer.py`: Core pipeline logic and CLI
- `route.py`: Simple route-style wrapper/demo for server integration
- `.env`: Local runtime configuration (not committed)
- `.env.example`: Shared config template for teammates
- `image/`: Input images
- `image_descriptions.json`: Output file (default)

## Prerequisites

- Python 3.10+ (3.11 recommended)
- xAI API key

## Library List

Runtime dependency:

- `xai-sdk`

Python standard library modules used:

- `argparse`
- `base64`
- `json`
- `mimetypes`
- `os`
- `pathlib`
- `typing`
- `urllib.parse`

## Install Procedure

### Windows (PowerShell)

1. Create venv:

```powershell
python -m venv .venv
```

2. Activate venv:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux/macOS (bash/zsh)

1. Create venv:

```bash
python3 -m venv .venv
```

2. Activate venv:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment Setup

1. Copy the example env file:

```bash
cp .env.example .env
```

On Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
```

2. Edit `.env` and set your values:

```dotenv
XAI_API_KEY=your_xai_api_key_here
XAI_BASE_URL=https://api.x.ai/v1
XAI_MODEL=grok-4-1-fast-non-reasoning
XAI_IMAGE_DETAIL=high
XAI_TIMEOUT=3600

IMAGE_INPUT=./image
IMAGE_OUTPUT=./image_descriptions.json
```

## How to Run

### Core script (single image)

```powershell
python grok_image_describer.py --image Acura_005.jpg
```

### Core script (all images)

```powershell
python grok_image_describer.py --all
```

### Route demo

```powershell
python route.py
```

## Server-Style Route Calls

### 1) By file name

```python
import route
result = route.describe_route({"image": "Acura_005.jpg"})
```

### 2) By file path

```python
import route
result = route.describe_route({"path": "./image/Acura_004.jpg"})
```

### 3) By directory path (all)

```python
import route
result = route.describe_route({"path": "./image", "run_all": True})
```

### 4) By directory path (first image only)

```python
import route
result = route.describe_route({"path": "./image", "run_all": False})
```

## Output Format

Example structure in `image_descriptions.json`:

```json
{
	"Acura_005.jpg": {
		"search_text": "...",
		"structured": {
			"long_description": "...",
			"short_caption": "...",
			"vehicle": {
				"make": "Acura",
				"model_guess": "...",
				"body_style": "...",
				"color": "..."
			},
			"scene": {
				"environment": "...",
				"lighting": "...",
				"camera_view": "...",
				"motion": "..."
			},
			"visual_attributes": ["..."],
			"query_phrases": ["..."]
		}
	}
}
```

## Notes

- Keep `.env` private; do not commit real API keys.
- If your key was exposed, rotate it in xAI console.
- `.env` is ignored by git in this project.