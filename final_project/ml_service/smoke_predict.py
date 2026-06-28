from __future__ import annotations

import json
from pathlib import Path

from .f18_predictor import DEFAULT_ARTIFACT_DIR, F18Predictor


def main() -> int:
    sample_path = DEFAULT_ARTIFACT_DIR / "sample_input.json"
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    payload.setdefault("interval_pct", 0.221796)
    payload.setdefault("interval_basis", "global_p95")
    predictor = F18Predictor(DEFAULT_ARTIFACT_DIR)
    print(json.dumps(predictor.predict_payload(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

