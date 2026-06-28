# F18 Prediction Service

This package loads the persisted `canonical_F18_reference_huber_010` Keras
artifact and exposes a minimal FastAPI prediction endpoint.

## Train and package artifact

```bash
cd /Users/gwongwangjae/goorm-ai-language-course/final_project
F18_ARTIFACT_FORCE=1 /Users/gwongwangjae/.Trash/miniforge3/bin/python3.10 scripts/train_f18_canonical_model_artifact.py
```

## Smoke predict

```bash
cd /Users/gwongwangjae/goorm-ai-language-course/final_project
/Users/gwongwangjae/.Trash/miniforge3/bin/python3.10 -m ml_service.smoke_predict
```

## Serve

```bash
cd /Users/gwongwangjae/goorm-ai-language-course/final_project
F18_ARTIFACT_DIR=models/f18_canonical_huber_010 uvicorn ml_service.main:app --host 127.0.0.1 --port 8001
```

The service expects already materialized F18 features. Spring Boot should keep
public API formatting, Redis caching, and unit conversion, while this service
only owns model loading and prediction.
