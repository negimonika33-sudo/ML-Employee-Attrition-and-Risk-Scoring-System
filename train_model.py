"""Train required models and generate all reusable project artifacts."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from workforce_assistant.data import load_dataset  # noqa: E402
from workforce_assistant.reports import write_reports  # noqa: E402
from workforce_assistant.training import save_model_bundle, train_models  # noqa: E402


def main() -> int:
    data, quality = load_dataset(PROJECT_DIR / "data" / "Palo Alto Networks.csv")
    bundle = train_models(data)
    artifact_directory = PROJECT_DIR / "artifacts"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    save_model_bundle(bundle, artifact_directory / "model_bundle.joblib")
    bundle["metrics"].to_csv(artifact_directory / "model_metrics.csv", index=False)
    bundle["feature_importance"].to_csv(
        artifact_directory / "feature_importance.csv", index=False
    )
    bundle["risk_scores"].to_csv(
        artifact_directory / "employee_risk_scores.csv", index=False
    )
    write_reports(data, bundle, PROJECT_DIR / "reports")

    print(f"Validated {quality['rows']:,} rows with {quality['missing_cells']} missing cells.")
    print(f"Selected model: {bundle['model_name']}")
    print(bundle["metrics"].to_string(index=False))
    print("Generated model artifacts and required reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

