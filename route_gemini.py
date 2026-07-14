import json
from typing import Any

from gemini_image_describer import run_pipeline
from route_common import dispatch_route


def describe_route(payload: dict[str, Any]) -> dict[str, Any]:
    return dispatch_route(payload, run_pipeline)


def _demo() -> None:
    print("Demo: process one image by name")
    one = describe_route({"image": "Acura_005.jpg"})
    print(json.dumps(one, indent=2, ensure_ascii=False))

    print("\nDemo: process one image by full file path")
    one_path = describe_route({"path": "./image/Acura_004.jpg"})
    print(json.dumps(one_path, indent=2, ensure_ascii=False))

    print("\nDemo: process all images from a directory path")
    all_result = describe_route({"path": "./image"})
    print(json.dumps(all_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _demo()
