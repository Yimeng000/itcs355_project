"""Update the cloud service using a registered model's deployment receipt."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from src import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--service", required=True)
    args = parser.parse_args()

    cfg = config.load(strict=False)
    if cfg.provider != "gcp":
        raise RuntimeError("This deployment adapter supports gcp")
    if cfg.project_id == "unset" or cfg.region == "unset":
        raise RuntimeError("Project and region must be configured")
    if "@sha256:" not in args.image:
        raise RuntimeError("Deployment requires an image digest")

    receipt = json.loads(
        (config.REPO_ROOT / "reports/registration.json").read_text()
    )
    variables = {
        "GOOGLE_CLOUD_PROJECT": cfg.project_id,
        "MODEL_ARTIFACT_URI": receipt["artifact_uri"] + "/model.joblib",
        "MODEL_VERSION": str(receipt["version_id"]),
    }

    result = subprocess.run(
        [
            "gcloud", "run", "services", "update", args.service,
            "--project", cfg.project_id,
            "--region", cfg.region,
            "--image", args.image,
            "--update-env-vars",
            ",".join(f"{key}={value}" for key, value in variables.items()),
            "--quiet",
            "--format=json",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    service = json.loads(result.stdout)
    url = service["status"]["url"]
    print("Service URL:", url)

    if os.getenv("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"url={url}\n")

    report = config.REPO_ROOT / "reports/deployment.json"
    report.write_text(json.dumps({
        "service_url": url,
        "image": args.image,
        "model_resource": receipt["model_resource"],
        "model_version": receipt["version_id"],
        "artifact_uri": receipt["artifact_uri"],
        "code_commit": os.getenv("GITHUB_SHA", "local"),
    }, indent=2))


if __name__ == "__main__":
    main()
