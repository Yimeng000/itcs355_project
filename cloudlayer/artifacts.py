"""Download model artifacts through the cloud-specific adapter."""
from pathlib import Path
from urllib.parse import urlparse


def download_model(uri: str, destination: Path) -> None:
    parsed = urlparse(uri)
    object_name = parsed.path.lstrip("/")
    if parsed.scheme != "gs" or not parsed.netloc or not object_name:
        raise ValueError("Invalid model artifact URI")

    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(parsed.netloc).blob(object_name)
    blob.download_to_filename(str(destination))
