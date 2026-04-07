import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib import error, parse, request


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_IMAGE_INPUT = "./image"
DEFAULT_IMAGE_OUTPUT = "./image_descriptions_gemini.json"
DEFAULT_DETAIL = "high"
VALID_DETAILS = {"auto", "low", "high"}
DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

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


def load_env_file(env_path: Path = Path(".env")) -> None:
    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe images with Gemini vision and append output to JSON."
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
        help="Gemini model name to use for image description.",
    )
    parser.add_argument(
        "--detail",
        choices=["auto", "low", "high"],
        default=None,
        help="Vision detail preference for prompt guidance.",
    )
    return parser.parse_args()


def get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set GEMINI_API_KEY in your environment.")
    return api_key


def get_timeout_seconds() -> float:
    raw = os.environ.get("GEMINI_TIMEOUT", "3600").strip()
    try:
        return float(raw)
    except ValueError:
        return 3600.0


def get_api_base() -> str:
    return os.environ.get("GEMINI_API_BASE", DEFAULT_API_BASE).rstrip("/")


def resolve_images(image_dir: Path, specific_image: str | None, run_all: bool) -> list[Path]:
    if specific_image:
        image_path = Path(specific_image)
        if not image_path.is_absolute():
            candidate = image_dir / specific_image
            image_path = candidate if candidate.exists() else image_path
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return [image_path]

    candidates = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")
    return candidates if run_all else [candidates[0]]


def _extract_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _safe_parse_json(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(_extract_json_object(text))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def _clean_structured(structured: dict[str, Any] | None) -> dict[str, Any] | None:
    if not structured:
        return None
    cleaned = dict(structured)
    cleaned.pop("confidence", None)
    return cleaned


def _build_search_text(structured: dict[str, Any], fallback_desc: str) -> str:
    if not structured:
        return fallback_desc

    vehicle = structured.get("vehicle", {}) if isinstance(structured.get("vehicle"), dict) else {}
    scene = structured.get("scene", {}) if isinstance(structured.get("scene"), dict) else {}
    attrs = structured.get("visual_attributes", [])
    queries = structured.get("query_phrases", [])

    parts: list[str] = []
    for value in [
        structured.get("short_caption"),
        structured.get("long_description"),
        vehicle.get("make"),
        vehicle.get("model_guess"),
        vehicle.get("body_style"),
        vehicle.get("color"),
        scene.get("environment"),
        scene.get("lighting"),
        scene.get("camera_view"),
        scene.get("motion"),
    ]:
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    if isinstance(attrs, list):
        parts.extend(str(x).strip() for x in attrs if str(x).strip())
    if isinstance(queries, list):
        parts.extend(str(x).strip() for x in queries if str(x).strip())

    return " | ".join(parts) if parts else fallback_desc


def _read_image_parts(image_path: Path) -> tuple[str, str]:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return mime_type, encoded


def _extract_text_from_gemini_response(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""

    first = candidates[0]
    if not isinstance(first, dict):
        return ""

    content = first.get("content")
    if not isinstance(content, dict):
        return ""

    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""

    text_chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            value = part.get("text")
            if isinstance(value, str) and value.strip():
                text_chunks.append(value.strip())

    return "\n".join(text_chunks).strip()


def describe_image(api_key: str, model: str, image_path: Path, detail: str) -> dict[str, Any]:
    mime_type, encoded_image = _read_image_parts(image_path)
    endpoint = f"{get_api_base()}/models/{model}:generateContent"
    url = f"{endpoint}?{parse.urlencode({'key': api_key})}"

    body = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}],
        },
        "generation_config": {
            "response_mime_type": "application/json",
        },
        "contents": [
            {
                "parts": [
                    {"text": f"{USER_PROMPT}\nVision detail preference: {detail}."},
                    {"inline_data": {"mime_type": mime_type, "data": encoded_image}},
                ]
            }
        ],
    }

    req = request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=get_timeout_seconds()) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed: HTTP {exc.code} - {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc

    payload = json.loads(raw)
    raw_desc = _extract_text_from_gemini_response(payload)
    structured = _clean_structured(_safe_parse_json(raw_desc))
    search_text = _build_search_text(structured or {}, raw_desc)

    return {
        "search_text": search_text,
        "structured": structured,
    }


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
        str(image_output) if image_output is not None else os.environ.get("GEMINI_IMAGE_OUTPUT", DEFAULT_IMAGE_OUTPUT)
    )
    model_name = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    detail_level = (detail or os.environ.get("GEMINI_IMAGE_DETAIL") or DEFAULT_DETAIL).lower().strip()
    if detail_level not in VALID_DETAILS:
        detail_level = DEFAULT_DETAIL

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
    records = load_existing_json(output_json)

    processed: list[str] = []
    for img_path in images:
        print(f"Processing: {img_path.name}")
        result = describe_image(api_key, model_name, img_path, detail_level)
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


def load_existing_json(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def save_json(path: Path, data: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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