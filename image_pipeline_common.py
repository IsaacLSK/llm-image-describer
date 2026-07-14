"""Shared utilities for image-description providers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
VALID_DETAILS = frozenset({"auto", "low", "high"})


def load_env_file(env_path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs without overriding the process environment."""
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def normalize_detail(value: str | None, default: str = "high") -> str:
    detail = (value or default).strip().lower()
    return detail if detail in VALID_DETAILS else default


def resolve_images(
    image_dir: Path,
    specific_image: str | None,
    run_all: bool,
) -> list[Path]:
    """Resolve either one explicitly selected image or images from a directory."""
    if specific_image:
        requested = Path(specific_image).expanduser()
        if not requested.is_absolute():
            in_image_dir = image_dir / requested
            requested = in_image_dir if in_image_dir.exists() else requested
        requested = requested.resolve()
        if not requested.is_file():
            raise FileNotFoundError(f"Image not found: {requested}")
        if requested.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported image type '{requested.suffix}'. Supported: {supported}")
        return [requested]

    candidates = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")
    return candidates if run_all else candidates[:1]


def safe_parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a model response, tolerating a Markdown fence or surrounding prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    candidate = cleaned[start : end + 1] if 0 <= start < end else cleaned
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    parsed.pop("confidence", None)
    return parsed


def build_search_text(structured: dict[str, Any] | None, fallback: str) -> str:
    """Flatten useful structured fields into a deduplicated search document."""
    if not structured:
        return fallback.strip()

    vehicle = structured.get("vehicle")
    scene = structured.get("scene")
    vehicle = vehicle if isinstance(vehicle, dict) else {}
    scene = scene if isinstance(scene, dict) else {}

    values: list[Any] = [
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
    ]
    for field in ("visual_attributes", "query_phrases"):
        items = structured.get(field)
        if isinstance(items, list):
            values.extend(items)

    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        part = value.strip()
        normalized = part.casefold()
        if part and normalized not in seen:
            seen.add(normalized)
            parts.append(part)
    return " | ".join(parts) or fallback.strip()


def parse_model_description(raw_description: str) -> dict[str, Any]:
    raw_description = raw_description.strip()
    if not raw_description:
        raise RuntimeError("The model returned an empty description")
    structured = safe_parse_json_object(raw_description)
    return {
        "search_text": build_search_text(structured, raw_description),
        "structured": structured,
    }


def load_existing_json(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Output file contains invalid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Output file must contain a JSON object: {path}")
    return data


def save_json(path: Path, data: dict[str, dict[str, Any]]) -> None:
    """Atomically replace the output file so interruptions cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
