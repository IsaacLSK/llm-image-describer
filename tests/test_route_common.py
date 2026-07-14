import tempfile
import unittest
from pathlib import Path
from typing import Any

from route_common import dispatch_route


class RouteCommonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def runner(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"processed": ["car.jpg"], "count": 1, "output_json": "/tmp/out.json"}

    def test_directory_defaults_to_all_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = dispatch_route({"path": directory}, self.runner)
        self.assertTrue(result["ok"])
        self.assertTrue(self.calls[0]["run_all"])

    def test_explicit_false_processes_first_directory_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dispatch_route({"path": directory, "run_all": False}, self.runner)
        self.assertFalse(self.calls[0]["run_all"])

    def test_file_path_selects_single_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "car.jpg"
            image.touch()
            result = dispatch_route({"path": str(image)}, self.runner)
        self.assertTrue(result["ok"])
        self.assertEqual(self.calls[0]["image"], str(image))
        self.assertFalse(self.calls[0]["run_all"])

    def test_runner_errors_are_returned(self) -> None:
        def failing_runner(**kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("provider unavailable")

        self.assertEqual(
            dispatch_route({"run_all": True}, failing_runner),
            {"ok": False, "error": "provider unavailable"},
        )


if __name__ == "__main__":
    unittest.main()
