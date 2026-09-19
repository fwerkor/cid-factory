# CID Factory

CID Factory is the training console for Continuous Interaction Diffusion.

It is intentionally a **frontend and control plane**, not a second training implementation. Model code, dataset semantics, optimizer behavior, Stage 0/A/B training, evaluation, and checkpoint formats remain owned by the main CID repository. Factory turns those existing entry points into a coherent WebUI, launches the exact main-repository commands, and visualizes the metrics that CID already writes.

## What is included

- polished responsive WebUI with dark and light themes;
- Stage 0, Stage A, and Stage B launch flows;
- exact command preview before launch;
- CUDA device selection and live GPU telemetry;
- persistent run history with the CID Git commit used for each launch;
- read-only asset catalog for local datasets, manifests, checkpoints, models, and run outputs;
- asset pickers in the launcher while preserving manual paths and Hugging Face model IDs;
- side-by-side comparison of 2–4 runs using native CID metrics and recorded launch settings;
- native CID training/validation metric visualization;
- process logs, command inspection, and stop control with persisted PID ownership verification;
- thin Python/FastAPI control plane;
- no duplicated CID model or training implementation.

## Architecture

    Browser
      |
      v
    CID Factory WebUI
      |
      v
    thin FastAPI control plane
      |
      +---- launch / observe processes
      +---- read native train_metrics*.jsonl
      +---- read validation_metrics.jsonl
      +---- discover local assets (read only)
      +---- compare recorded runs
      +---- query accelerator telemetry
      |
      v
    continuous-interaction-diffusion
      |
      +---- scripts/train_diffusion_base.py   (Stage 0)
      +---- cid train                         (Stage A)
      +---- cid train-full                    (Stage B)

Factory records the resolved command and current CID source commit for every run. If CID rejects an invalid configuration, that validation remains authoritative. Factory never loads checkpoint tensors just to populate the UI; asset discovery reads filenames and small JSON metadata only.

After a Factory restart, a persisted PID is considered controllable only when it belongs to the same host and still carries that run's `CID_FACTORY_RUN_ID` marker. This prevents a recycled OS PID from being signalled accidentally.

## Development

Requirements:

- Python 3.10+
- Node.js 20+
- a local checkout of the main CID repository

By default Factory looks for a sibling directory named continuous-interaction-diffusion. Override it with:

    export CID_FACTORY_CID_REPO=/path/to/continuous-interaction-diffusion

Factory itself stays lightweight. It separately discovers a Python environment with PyTorch for
the actual CID process. Pin that environment explicitly when needed:

    export CID_FACTORY_CID_PYTHON=/path/to/cid-env/bin/python

The Assets workspace scans a small default set of CID-adjacent directories. Provide additional or replacement roots as an OS path-separated list:

    export CID_FACTORY_ASSET_ROOTS=/data/cid:/models/cid:/scratch/cid-runs

Install the control plane:

    python -m venv .venv
    . .venv/bin/activate
    pip install -e '.[dev]'

Build the frontend:

    cd web
    npm install
    npm run build
    cd ..

Run the integrated application:

    cid-factory --host 0.0.0.0 --port 7860

The same paths can be provided without environment variables:

    cid-factory \
      --cid-repo /path/to/continuous-interaction-diffusion \
      --cid-python /path/to/cid-env/bin/python \
      --host 0.0.0.0

For frontend development, run the API on port 7860 and Vite separately:

    cid-factory --reload
    cd web && npm run dev

Vite proxies /api to the local Factory API.

## State

Factory stores only frontend/control-plane state. The default location is:

    ~/.local/share/cid-factory

Override it with CID_FACTORY_STATE_DIR or --state-dir. Training outputs remain wherever the main CID command's --output-dir points.

## Design principle

If a behavior can affect model semantics, it belongs in the main CID repository. Factory may expose, validate, preview, launch, stop, and visualize that behavior, but it should not silently reimplement it.

Apache-2.0.
