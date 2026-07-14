import json
import tempfile
import unittest
from pathlib import Path

from image_pipeline_common import (
    build_search_text,
    load_existing_json,
    normalize_detail,
    parse_model_description,
    resolve_images,
    save_json,
)


class ImagePipelineCommonTests(unittest.TestCase):
    def test_normalize_detail_is_case_insensitive_and_has_fallback(self) -> None:
        self.assertEqual(normalize_detail(" LOW "), "low")
        self.assertEqual(normalize_detail("invalid"), "high")

    def test_parse_model_description_handles_fenced_json(self) -> None:
        result = parse_model_description(
            '```json\n{"short_caption":"Blue coupe","confidence":0.8}\n```'
        )
        self.assertEqual(result["structured"], {"short_caption": "Blue coupe"})
        self.assertEqual(result["search_text"], "Blue coupe")

    def test_build_search_text_deduplicates_case_insensitively(self) -> None:
        structured = {
            "short_caption": "Red sedan",
            "visual_attributes": ["red sedan", "alloy wheels"],
        }
        self.assertEqual(build_search_text(structured, "fallback"), "Red sedan | alloy wheels")

    def test_resolve_images_filters_and_sorts_supported_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.png").touch()
            (root / "a.jpg").touch()
            (root / "notes.txt").touch()
            self.assertEqual(
                [path.name for path in resolve_images(root, None, True)],
                ["a.jpg", "b.png"],
            )
            self.assertEqual(resolve_images(root, None, False)[0].name, "a.jpg")

    def test_resolve_images_rejects_explicit_unsupported_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsupported = root / "image.gif"
            unsupported.touch()
            with self.assertRaises(ValueError):
                resolve_images(root, str(unsupported), False)

    def test_json_round_trip_and_invalid_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "output.json"
            expected = {"car.jpg": {"search_text": "car", "structured": None}}
            save_json(output, expected)
            self.assertEqual(load_existing_json(output), expected)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), expected)

            output.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_existing_json(output)


if __name__ == "__main__":
    unittest.main()
