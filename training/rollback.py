from pathlib import Path
import json
import shutil


MODELS_DIR = Path("models")

PRODUCTION_MODEL = MODELS_DIR / "ticket_classifier.joblib"
PRODUCTION_METADATA = MODELS_DIR / "production_metadata.json"

ROLLBACK_MODEL = MODELS_DIR / "ticket_classifier_v1.joblib"


def load_metadata(path):
    with open(path, "r") as f:
        return json.load(f)


def rollback():
    if not PRODUCTION_MODEL.exists():
        raise FileNotFoundError(
            "Production model does not exist."
        )

    if not PRODUCTION_METADATA.exists():
        raise FileNotFoundError(
            "Production metadata does not exist."
        )

    if not ROLLBACK_MODEL.exists():
        raise FileNotFoundError(
            "v1 model backup does not exist."
        )

    rollback_metadata_path = (
        MODELS_DIR / "production_metadata_v1.json"
    )

    if not rollback_metadata_path.exists():
        raise FileNotFoundError(
            "v1 metadata backup does not exist."
        )

    rollback_metadata = load_metadata(
        rollback_metadata_path
    )

    shutil.copy2(
        ROLLBACK_MODEL,
        PRODUCTION_MODEL,
    )

    shutil.copy2(
        rollback_metadata_path,
        PRODUCTION_METADATA,
    )

    print(
        f"Rolled back production to "
        f"{rollback_metadata['model_version']}."
    )

    print(
        f"Macro F1: "
        f"{rollback_metadata['macro_f1']}"
    )


if __name__ == "__main__":
    rollback()