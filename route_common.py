"""Provider-independent request routing for image-description pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


PipelineRunner = Callable[..., dict[str, Any]]


def dispatch_route(payload: dict[str, Any], runner: PipelineRunner) -> dict[str, Any]:
    """Translate a route-style payload into a pipeline invocation."""
    try:
        path_value = payload.get("path")
        image = payload.get("image")
        run_all = bool(payload.get("run_all", False))
        image_input = payload.get("image_input")
        options = {
            "image_output": payload.get("image_output"),
            "model": payload.get("model"),
            "detail": payload.get("detail"),
        }

        if path_value:
            path = Path(path_value).expanduser()
            if path.is_dir():
                # A directory path means all images unless run_all=false is explicit.
                should_run_all = payload.get("run_all", True) is not False
                result = runner(
                    run_all=should_run_all,
                    image_input=str(path),
                    **options,
                )
                return {"ok": True, **result}
            if path.is_file():
                result = runner(
                    image=str(path),
                    run_all=False,
                    image_input=str(path.parent),
                    **options,
                )
                return {"ok": True, **result}
            return {"ok": False, "error": "Provided 'path' does not exist"}

        if image:
            result = runner(
                image=str(image),
                run_all=False,
                image_input=image_input,
                **options,
            )
            return {"ok": True, **result}

        if run_all:
            result = runner(run_all=True, image_input=image_input, **options)
            return {"ok": True, **result}

        return {
            "ok": False,
            "error": "Provide one of: path (file/dir), image (name/path), or run_all=true",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
