# Trust Verification Layer

This package provides a minimal FastAPI service that tracks integrity proofs for the static CSV datasets currently used across the project.

## Key commands

```bash
# (Optional) install dependencies
pip install -r trust_layer/requirements.txt

# Rebuild hashes / simulated proofs after CSVs change
python -m trust_layer.generate_proofs

# Start the verification API (http://localhost:8000)
uvicorn trust_layer.api:app --reload
```

## API surface

- `GET /trust/datasets` – returns the cached registry of dataset hashes, EigenLayer
  simulation notes, and mock ZKP status.
- `POST /trust/verify` – recomputes hashes on demand and updates `trust_proofs.json`.
- `GET /health` – liveness endpoint for uptime checks.

The generated `trust_layer/trust_proofs.json` file can be served to the Next.js
frontend or bundled as part of the deployment artifacts. Later, when new ML
artifacts or automated datasets are added, simply extend `trust_layer/config.py`
with their paths and rerun the generator.
