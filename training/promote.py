from pathlib import Path
import json
import shutil


MODELS_DIR = Path("models")

PRODUCTION_MODEL = MODELS_DIR / "ticket_classifier.joblib"
CANDIDATE_MODEL = MODELS_DIR / "ticket_classifier_candidate.joblib"

PRODUCTION_METADATA = MODELS_DIR / "production_metadata.json"
CANDIDATE_METADATA = MODELS_DIR / "candidate_metadata.json"


def load_metadata(path):
    with open(path, "r") as f:
        return json.load(f)


def promote():
    if not CANDIDATE_MODEL.exists():
        raise FileNotFoundError(
            "Candidate model does not exist."
        )

    if not PRODUCTION_MODEL.exists():
        raise FileNotFoundError(
            "Production model does not exist."
        )

    if not CANDIDATE_METADATA.exists():
        raise FileNotFoundError(
            "Candidate metadata does not exist."
        )

    if not PRODUCTION_METADATA.exists():
        raise FileNotFoundError(
            "Production metadata does not exist."
        )

    candidate = load_metadata(CANDIDATE_METADATA)
    production = load_metadata(PRODUCTION_METADATA)

    candidate_f1 = candidate["macro_f1"]
    production_f1 = production["macro_f1"]

    print(f"Production F1: {production_f1}")
    print(f"Candidate F1:  {candidate_f1}")

    if candidate_f1 <= production_f1:
        print("\nPromotion rejected.")
        print("Production model remains unchanged.")
        return

    print("\nPromotion approved.")

    # Backup current production model
    production_version = production["model_version"]

    backup_model = (
        MODELS_DIR
        / f"ticket_classifier_{production_version}.joblib"
    )

    backup_metadata = (
        MODELS_DIR
        / f"production_metadata_{production_version}.json"
    )

    # Backup current production model
    shutil.copy2(
        PRODUCTION_MODEL,
        backup_model,
    )

    # Backup current production metadata
    shutil.copy2(
        PRODUCTION_METADATA,
        backup_metadata,
    )

    # Promote candidate
    shutil.copy2(
        CANDIDATE_MODEL,
        PRODUCTION_MODEL,
    )

    # Update production metadata
    new_version_number = int(
        production_version.replace("v", "")
    ) + 1

    promoted_metadata = candidate.copy()
    promoted_metadata["model_version"] = (
        f"v{new_version_number}"
    )

    with open(PRODUCTION_METADATA, "w") as f:
        json.dump(
            promoted_metadata,
            f,
            indent=4,
        )

    print(
        f"\nCandidate promoted to production "
        f"as v{new_version_number}."
    )
if __name__ == "__main__":
    promote()