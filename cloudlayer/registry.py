"""Cloud-specific artifact upload and model registration."""
from __future__ import annotations

import json
from urllib.parse import urlparse


def register_model(cfg, files, manifest, image, parent_model):
    if cfg.provider != "gcp":
        raise RuntimeError("This registry adapter currently supports gcp")
    if cfg.project_id == "unset" or cfg.region == "unset":
        raise RuntimeError("Configure PROJECT_ID and REGION")

    uri = urlparse(cfg.blob_uri)
    if uri.scheme != "gs" or not uri.netloc:
        raise RuntimeError("Configure BLOB_URI with a storage location")

    from google.cloud import aiplatform, storage

    suffix = (
        f"models/{manifest['model_sha256'][:16]}/"
        f"{manifest['training_run_id']}"
    )
    prefix = "/".join(part for part in (uri.path.strip("/"), suffix) if part)
    artifact_uri = f"gs://{uri.netloc}/{prefix}"

    bucket = storage.Client(project=cfg.project_id).bucket(uri.netloc)
    for name, path in files.items():
        bucket.blob(f"{prefix}/{name}").upload_from_filename(str(path))

    aiplatform.init(project=cfg.project_id, location=cfg.region)
    description = json.dumps(manifest, sort_keys=True)
    model = aiplatform.Model.upload(
        display_name=cfg.model_registry_name,
        parent_model=parent_model,
        is_default_version=False,
        version_aliases=["candidate"],
        artifact_uri=artifact_uri,
        serving_container_image_uri=image,
        serving_container_ports=[8080],
        serving_container_health_route="/ready",
        serving_container_predict_route="/predict",
        serving_container_environment_variables={
            "MODEL_ARTIFACT_URI": f"{artifact_uri}/model.joblib",
            "MODEL_VERSION": manifest["model_sha256"][:16],
        },
        version_description=description,
        labels={
            "course": "itcs355",
            "training_commit": manifest["training_git_commit"][:40],
            "training_run": manifest["training_run_id"],
            "model_hash": manifest["model_sha256"][:16],
        },
    )
    return {
        "model_resource": model.resource_name,
        "version_id": model.version_id,
        "artifact_uri": artifact_uri,
        "serving_image": image,
        "lineage": manifest,
    }
