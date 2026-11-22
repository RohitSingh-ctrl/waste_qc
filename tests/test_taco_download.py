import http.client
import io
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.data.taco import _fetch, download_taco


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (4, 4), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_download_taco_replaces_corrupt_existing_image(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "file_name": "batch/image.jpg",
                        "flickr_url": "https://example.test/image.jpg",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    image_path = tmp_path / "batch" / "image.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"")

    with patch("src.data.taco.urllib.request.urlopen", return_value=io.BytesIO(_jpeg_bytes())):
        result = download_taco(tmp_path, "https://example.test/annotations.json")

    assert result == annotations
    assert Image.open(image_path).size == (4, 4)


def test_fetch_does_not_keep_invalid_image_response(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"

    with patch("src.data.taco.urllib.request.urlopen", return_value=io.BytesIO(b"not an image")):
        downloaded = _fetch("https://example.test/image.jpg", image_path, validate_image=True)

    assert not downloaded
    assert not image_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_fetch_cleans_up_after_interrupted_response(tmp_path: Path) -> None:
    class InterruptedResponse:
        def __init__(self) -> None:
            self.read_count = 0

        def __enter__(self) -> "InterruptedResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            self.read_count += 1
            if self.read_count == 1:
                return b"partial"
            raise http.client.IncompleteRead(b"partial")

    image_path = tmp_path / "image.jpg"

    with patch(
        "src.data.taco.urllib.request.urlopen",
        return_value=InterruptedResponse(),
    ):
        downloaded = _fetch("https://example.test/image.jpg", image_path, validate_image=True)

    assert not downloaded
    assert not image_path.exists()
    assert list(tmp_path.iterdir()) == []
