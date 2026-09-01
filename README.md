# AttnLab

AttnLab is a visual execution workspace for a deterministic toy
Multi-Head Attention implementation.

Supported attention paths:

- MHA
- MQA
- GQA
- MHA with RoPE
- MLA with a persistent low-rank latent cache
- KDA with a persistent recurrent delta-rule state
- CSA with compressed Top-K routing
- HCA with hierarchical compressed routing

## Run locally

Backend:

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open <http://127.0.0.1:5173>.

## Verify

```bash
PYTHONPATH=backend .venv/bin/python -m unittest discover -s backend/tests
cd frontend && npm run build
```
