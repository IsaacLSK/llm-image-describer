from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from image_pipeline_common import (
    load_env_file,
    load_existing_json,
    normalize_detail,
    parse_model_description,
    resolve_images,
    save_json,
)


DEFAULT_MODEL = "grok-4-1-fast-non-reasoning"
DEFAULT_IMAGE_INPUT = "./image"
DEFAULT_IMAGE_OUTPUT = "./image_descriptions.json"
DEFAULT_DETAIL = "high"

SYSTEM_PROMPT = (
    "You are an automotive vision labeling assistant. "
    "Generate descriptions optimized for semantic/vector search. "
    "Be factual, avoid guessing beyond visible evidence, and provide concise normalized attributes."
)

USER_PROMPT = (
    "Analyze this image and return JSON only. No markdown, no extra text. "
    "Use this schema exactly:\n"
    "{\n"
    "  \"long_description\": string,\n"
    "  \"short_caption\": string,\n"
    "  \"vehicle\": {\n"
    "    \"make\": string,\n"
    "    \"model_guess\": string,\n"
    "    \"body_style\": string,\n"
    "    \"color\": string\n"
    "  },\n"
    "  \"scene\": {\n"
    "    \"environment\": string,\n"
    "    \"lighting\": string,\n"
    "    \"camera_view\": string,\n"
    "    \"motion\": string\n"
    "  },\n"
    "  \"visual_attributes\": [string],\n"
    "  \"query_phrases\": [string]\n"
    "}\n"
    "Rules: query_phrases should contain natural user-like search queries such as color + body style + view + scene."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe images with Grok (xAI) and append output to JSON."
    )
    parser.add_argument(
        "--image-input",
        "--image-dir",
        dest="image_input",
        default=None,
        help="Directory containing images.",
    )
    parser.add_argument(
        "--image-output",
        "--output-json",
        dest="image_output",
        default=None,
        help="Path to JSON output file.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Specific image filename or path. If omitted, first image is used unless --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all supported images in --image-dir.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="xAI model name to use for image description.",
    )
    parser.add_argument(
        "--detail",
        choices=["auto", "low", "high"],
        default=None,
        help="Vision detail level for image parsing.",
    )
    return parser.parse_args()


def get_api_key() -> str:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set XAI_API_KEY in your environment.")
    return api_key


def get_api_host() -> str:
    api_host = os.environ.get("XAI_API_HOST")
    if api_host:
        return api_host

    base_url = os.environ.get("XAI_BASE_URL", "").strip()
    if not base_url:
        return "api.x.ai"

    if "://" not in base_url:
        return base_url

    parsed = urlparse(base_url)
    return parsed.netloc or "api.x.ai"


def get_timeout_seconds() -> float:
    raw = os.environ.get("XAI_TIMEOUT", "3600").strip()
    try:
        return float(raw)
    except ValueError:
        return 3600.0


def to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def describe_image(client: Any, model: str, image_path: Path, detail: str) -> dict[str, Any]:
    from xai_sdk.chat import image, system, user

    data_url = to_data_url(image_path)

    chat = client.chat.create(model=model)
    chat.append(system(SYSTEM_PROMPT))
    chat.append(
        user(
            USER_PROMPT,
            image(data_url, detail=detail),
        )
    )
    response = chat.sample()
    return parse_model_description(response.content)


def run_pipeline(
    *,
    image: str | None = None,
    run_all: bool = False,
    image_input: str | Path | None = None,
    image_output: str | Path | None = None,
    model: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    load_env_file()

    image_dir_raw = str(image_input) if image_input is not None else os.environ.get("IMAGE_INPUT", DEFAULT_IMAGE_INPUT)
    output_json_raw = (
        str(image_output) if image_output is not None else os.environ.get("IMAGE_OUTPUT", DEFAULT_IMAGE_OUTPUT)
    )
    model_name = model or os.environ.get("XAI_MODEL") or DEFAULT_MODEL
    detail_level = normalize_detail(detail or os.environ.get("XAI_IMAGE_DETAIL"), DEFAULT_DETAIL)

    input_path = Path(image_dir_raw).resolve()
    # Support file-path input directly: treat it as single-image mode.
    if input_path.exists() and input_path.is_file():
        if image is None:
            image = str(input_path)
        run_all = False
        image_dir = input_path.parent
    else:
        image_dir = input_path

    output_json = Path(output_json_raw).resolve()

    if not image_dir.exists() or not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_dir}")

    api_key = get_api_key()
    images = resolve_images(image_dir, image, run_all)

    try:
        from xai_sdk import Client
    except ImportError as exc:
        raise RuntimeError(
            "The Grok provider requires xai-sdk. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    client = Client(api_key=api_key, api_host=get_api_host(), timeout=get_timeout_seconds())
    records = load_existing_json(output_json)

    processed: list[str] = []
    for img_path in images:
        print(f"Processing: {img_path.name}")
        result = describe_image(client, model_name, img_path, detail_level)
        records[img_path.name] = {
            "search_text": result["search_text"],
            "structured": result["structured"],
        }
        save_json(output_json, records)
        processed.append(img_path.name)
        print(f"Saved description for: {img_path.name}")

    print(f"Done. Output JSON: {output_json}")
    return {
        "processed": processed,
        "count": len(processed),
        "output_json": str(output_json),
    }


def main() -> None:
    args = parse_args()
    run_pipeline(
        image=args.image,
        run_all=args.all,
        image_input=args.image_input,
        image_output=args.image_output,
        model=args.model,
        detail=args.detail,
    )


if __name__ == "__main__":
    main()
